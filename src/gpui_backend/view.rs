use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use gpui_kit::prelude::*;
use gpui_kit::{
    Bounds, EventEmitter, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, PinchEvent,
    Pixels, Point, ScrollEvent, ScrollSource, StatefulInteractiveElement, Window, canvas, div, px,
};

use crate::geom::{Point as DataPoint, ScreenPoint, ScreenRect};
use crate::interaction::{
    HitRegion, pan_viewport, toggle_pin, zoom_factor_from_drag, zoom_to_rect, zoom_viewport,
};
use crate::plot::Plot;
use crate::transform::Transform;
use crate::view::{Range, Viewport};

use super::config::PlotViewConfig;
use super::constants::DOUBLE_CLICK_PIN_GRACE_MS;
use super::frame::build_frame;
use super::geometry::{distance_sq, normalized_rect};
use super::hover::{compute_hover_target, hover_target_within_threshold};
use super::link::{LinkBinding, PlotLinkGroup, PlotLinkOptions, ViewSyncKind};
use super::paint::{paint_frame, to_hsla};
use super::state::{ClickState, DragMode, DragState, PinToggle, PlotUiState};

/// Events emitted by [`PlotView`] when the user interacts with a chart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlotEvent {
    /// Pointer hover X position in data coordinates, or `None` when cleared.
    CursorX(Option<f64>),
    /// Primary click/tap X position in data coordinates.
    SelectedX(f64),
    /// Visible data viewport changed by pan, zoom, or reset (`None`).
    ViewportChanged(Option<Viewport>),
}

/// A GPUI view that renders a [`Plot`] with interactive controls.
///
/// This view handles pan/zoom/box-zoom, hover readouts, and pin interactions
/// while delegating data management to the underlying [`Plot`].
#[derive(Debug, Clone)]
pub struct PlotView {
    plot: Arc<RwLock<Plot>>,
    state: Arc<RwLock<PlotUiState>>,
    config: PlotViewConfig,
    link: Option<LinkBinding>,
}

impl EventEmitter<PlotEvent> for PlotView {}

impl PlotView {
    /// Create a new GPUI plot view for the given plot.
    ///
    /// Uses the default [`PlotViewConfig`].
    pub fn new(plot: Plot) -> Self {
        Self {
            plot: Arc::new(RwLock::new(plot)),
            state: Arc::new(RwLock::new(PlotUiState::default())),
            config: PlotViewConfig::default(),
            link: None,
        }
    }

    /// Create a new GPUI plot view with a custom configuration.
    pub fn with_config(plot: Plot, config: PlotViewConfig) -> Self {
        Self {
            plot: Arc::new(RwLock::new(plot)),
            state: Arc::new(RwLock::new(PlotUiState::default())),
            config,
            link: None,
        }
    }

    /// Attach this view to a multi-plot link group.
    ///
    /// Link groups synchronize viewport/cursor/brush state between views.
    pub fn with_link_group(mut self, group: PlotLinkGroup, options: PlotLinkOptions) -> Self {
        self.link = Some(LinkBinding {
            member_id: group.register_member(),
            group,
            options,
        });
        self
    }

    /// Get a handle for mutating the underlying plot.
    ///
    /// This is useful for streaming updates from async tasks.
    pub fn plot_handle(&self) -> PlotHandle {
        PlotHandle {
            plot: Arc::clone(&self.plot),
        }
    }

    /// Current window-relative view bounds after the latest layout pass.
    pub fn bounds(&self) -> Option<Bounds<Pixels>> {
        self.state.read().expect("plot state lock").view_bounds
    }

    /// Set or clear a persistent vertical marker at data coordinate `x`.
    pub fn set_selected_x(&mut self, x: Option<f64>, cx: &mut Context<Self>) {
        self.state.write().expect("plot state lock").selected_x = x.filter(|v| v.is_finite());
        cx.notify();
    }

    /// Show or hide the in-plot legend overlay.
    pub fn set_show_legend(&mut self, show: bool, cx: &mut Context<Self>) {
        self.config.show_legend = show;
        cx.notify();
    }

    /// Reset manual pan/zoom, notify any linked views, and emit [`PlotEvent::ViewportChanged`].
    pub fn reset_view(&mut self, cx: &mut Context<Self>) {
        {
            let mut state = self.state.write().expect("plot state lock");
            if let Ok(mut plot) = self.plot.write() {
                plot.reset_view();
            }
            state.linked_brush_x = None;
            state.clear_interaction();
            self.publish_reset_link();
            self.publish_brush_link(None);
        }
        cx.emit(PlotEvent::ViewportChanged(None));
        cx.notify();
    }

    /// Zoom the viewport by multiplicative scale factors (`> 1.0` zooms in) around `screen_pos`.
    pub fn zoom_by(
        &mut self,
        factor_x: f32,
        factor_y: f32,
        screen_pos: Option<Point<Pixels>>,
        cx: &mut Context<Self>,
    ) {
        if !factor_x.is_finite() || !factor_y.is_finite() || factor_x <= 0.0 || factor_y <= 0.0 {
            return;
        }
        let mut changed = None;
        {
            let mut state = self.state.write().expect("plot state lock");
            let Some(transform) = state.transform.clone() else {
                return;
            };
            let Some(rect) = state.plot_rect else {
                return;
            };
            if let Ok(mut plot) = self.plot.write()
                && let Some(viewport) = plot.viewport()
            {
                let center = screen_pos
                    .and_then(|pos| transform.screen_to_data(screen_point(pos)))
                    .unwrap_or_else(|| viewport.center());
                let next = zoom_viewport(
                    viewport,
                    center,
                    (1.0 / f64::from(factor_x)).clamp(0.1, 10.0),
                    (1.0 / f64::from(factor_y)).clamp(0.1, 10.0),
                );
                self.apply_manual_view_with_link(&mut plot, &mut state, rect, next);
                changed = Some(next);
            }
        }
        if let Some(next) = changed {
            cx.emit(PlotEvent::ViewportChanged(Some(next)));
            cx.notify();
        }
    }

    fn publish_manual_view_link(&self, viewport: Viewport) {
        let Some(link) = self.link.as_ref() else {
            return;
        };
        link.group.publish_manual_view(
            link.member_id,
            viewport,
            link.options.link_x,
            link.options.link_y,
        );
    }

    fn publish_reset_link(&self) {
        let Some(link) = self.link.as_ref() else {
            return;
        };
        if link.options.link_reset {
            link.group.publish_reset(link.member_id);
        }
    }

    fn publish_cursor_link(&self, x: Option<f64>) {
        let Some(link) = self.link.as_ref() else {
            return;
        };
        if link.options.link_cursor {
            link.group.publish_cursor_x(link.member_id, x);
        }
    }

    fn publish_brush_link(&self, x_range: Option<Range>) {
        let Some(link) = self.link.as_ref() else {
            return;
        };
        if link.options.link_brush {
            link.group.publish_brush_x(link.member_id, x_range);
        }
    }

    fn apply_manual_view_with_link(
        &self,
        plot: &mut Plot,
        state: &mut PlotUiState,
        rect: ScreenRect,
        viewport: Viewport,
    ) {
        apply_manual_view(plot, state, rect, viewport);
        state.linked_brush_x = None;
        self.publish_manual_view_link(viewport);
        self.publish_brush_link(None);
    }

    fn on_mouse_down(&mut self, ev: &MouseDownEvent, cx: &mut Context<Self>) {
        let pos = screen_point(ev.position);
        let mut state = self.state.write().expect("plot state lock");
        state.last_cursor = Some(pos);

        if let Some(series_id) = state.legend_hit(pos) {
            if ev.button == MouseButton::Left && ev.click_count == 1 {
                if let Ok(mut plot) = self.plot.write() {
                    if let Some(series) = plot
                        .series_mut()
                        .iter_mut()
                        .find(|series| series.id() == series_id)
                    {
                        series.set_visible(!series.is_visible());
                    }
                }
            }
            state.clear_interaction();
            state.hover = None;
            state.hover_target = None;
            cx.notify();
            return;
        }

        let region = state.regions.hit_test(pos);
        if ev.button == MouseButton::Left && ev.click_count >= 2 && region == HitRegion::Plot {
            let last_toggle = state.last_pin_toggle.take();
            if let Ok(mut plot) = self.plot.write() {
                if let Some(last_toggle) = last_toggle {
                    if last_toggle.at.elapsed() <= Duration::from_millis(DOUBLE_CLICK_PIN_GRACE_MS)
                        && distance_sq(last_toggle.screen_pos, pos)
                            <= self.config.pin_threshold_px.powi(2)
                    {
                        revert_pin_toggle(&mut plot, last_toggle);
                    }
                }
                plot.reset_view();
                state.linked_brush_x = None;
                self.publish_reset_link();
                self.publish_brush_link(None);
            }
            state.clear_interaction();
            drop(state);
            cx.emit(PlotEvent::ViewportChanged(None));
            cx.notify();
            return;
        }

        state.pending_click = Some(ClickState {
            region,
            button: ev.button,
        });

        match (ev.button, region) {
            (MouseButton::Left, HitRegion::XAxis) => {
                state.drag = Some(DragState::new(DragMode::ZoomX, pos, true));
            }
            (MouseButton::Left, HitRegion::YAxis) => {
                state.drag = Some(DragState::new(DragMode::ZoomY, pos, true));
            }
            (MouseButton::Left, HitRegion::Plot) => {
                state.drag = Some(DragState::new(DragMode::Pan, pos, false));
            }
            (MouseButton::Right, HitRegion::Plot) => {
                state.drag = Some(DragState::new(DragMode::ZoomRect, pos, true));
                state.selection_rect = Some(ScreenRect::new(pos, pos));
            }
            _ => {}
        }

        cx.notify();
    }

    fn on_mouse_move(&mut self, ev: &MouseMoveEvent, cx: &mut Context<Self>) {
        let pos = screen_point(ev.position);
        let cursor_x;
        let mut changed_view = None;
        {
            let mut state = self.state.write().expect("plot state lock");
            state.last_cursor = Some(pos);

            if state.legend_hit(pos).is_some() {
                state.hover = None;
            } else if state.regions.hit_test(pos) == HitRegion::Plot {
                state.hover = Some(pos);
            } else {
                state.hover = None;
            }
            let linked_cursor_x = state.hover.and_then(|_| {
                state
                    .transform
                    .as_ref()
                    .and_then(|transform| transform.screen_to_data(pos))
                    .map(|point| point.x)
            });
            self.publish_cursor_link(linked_cursor_x);
            cursor_x = linked_cursor_x;

            let Some(mut drag) = state.drag.clone() else {
                drop(state);
                cx.emit(PlotEvent::CursorX(cursor_x));
                cx.notify();
                return;
            };

            if !is_drag_button_held(drag.mode, ev.pressed_button) {
                state.clear_interaction();
                self.publish_cursor_link(None);
                drop(state);
                cx.emit(PlotEvent::CursorX(None));
                cx.notify();
                return;
            }

            let moved_sq = distance_sq(drag.start, pos);
            if !drag.active && moved_sq > self.config.drag_threshold_px.powi(2) {
                drag.active = true;
            }

            if !drag.active {
                state.drag = Some(drag);
                drop(state);
                cx.emit(PlotEvent::CursorX(cursor_x));
                cx.notify();
                return;
            }

            let delta = ScreenPoint::new(pos.x - drag.last.x, pos.y - drag.last.y);
            let plot_rect = state.plot_rect;
            let transform = state.transform.clone();

            match drag.mode {
                DragMode::Pan => {
                    if let (Some(rect), Some(transform)) = (plot_rect, transform) {
                        if let Ok(mut plot) = self.plot.write() {
                            if let Some(viewport) = plot.viewport() {
                                if let Some(next) = pan_viewport(viewport, delta, &transform) {
                                    self.apply_manual_view_with_link(
                                        &mut plot, &mut state, rect, next,
                                    );
                                    changed_view = Some(next);
                                }
                            }
                        }
                    }
                }
                DragMode::ZoomRect => {
                    state.selection_rect = Some(ScreenRect::new(drag.start, pos));
                }
                DragMode::ZoomX => {
                    if let (Some(rect), Some(transform)) = (plot_rect, transform) {
                        let axis_pixels = rect.width().max(1.0);
                        let factor = zoom_factor_from_drag(delta.x, axis_pixels);
                        if let Ok(mut plot) = self.plot.write() {
                            if let Some(viewport) = plot.viewport() {
                                let center = transform
                                    .screen_to_data(pos)
                                    .unwrap_or_else(|| viewport.x_center());
                                let next = zoom_viewport(viewport, center, factor, 1.0);
                                self.apply_manual_view_with_link(&mut plot, &mut state, rect, next);
                                changed_view = Some(next);
                            }
                        }
                    }
                }
                DragMode::ZoomY => {
                    if let (Some(rect), Some(transform)) = (plot_rect, transform) {
                        let axis_pixels = rect.height().max(1.0);
                        let factor = zoom_factor_from_drag(-delta.y, axis_pixels);
                        if let Ok(mut plot) = self.plot.write() {
                            if let Some(viewport) = plot.viewport() {
                                let center = transform
                                    .screen_to_data(pos)
                                    .unwrap_or_else(|| viewport.y_center());
                                let next = zoom_viewport(viewport, center, 1.0, factor);
                                self.apply_manual_view_with_link(&mut plot, &mut state, rect, next);
                                changed_view = Some(next);
                            }
                        }
                    }
                }
            }

            drag.last = pos;
            state.drag = Some(drag);
            state.pending_click = None;
        }

        cx.emit(PlotEvent::CursorX(cursor_x));
        if let Some(next) = changed_view {
            cx.emit(PlotEvent::ViewportChanged(Some(next)));
        }
        cx.notify();
    }

    fn on_hover_state_change(&mut self, hovered: bool, window: &Window, cx: &mut Context<Self>) {
        if hovered {
            return;
        }

        let cursor = screen_point(window.mouse_position());
        let mut state = self.state.write().expect("plot state lock");
        let still_inside = state.legend_hit(cursor).is_some()
            || state.regions.hit_test(cursor) != HitRegion::Outside;
        if still_inside {
            return;
        }

        let changed = state.hover.take().is_some() || state.hover_target.take().is_some();
        state.last_cursor = None;
        drop(state);

        self.publish_cursor_link(None);
        if changed {
            cx.emit(PlotEvent::CursorX(None));
            cx.notify();
        }
    }

    fn on_mouse_up(&mut self, ev: &MouseUpEvent, cx: &mut Context<Self>) {
        let pos = screen_point(ev.position);
        let mut selected_x = None;
        let mut changed_view = None;
        {
            let mut state = self.state.write().expect("plot state lock");
            let drag = state.drag.clone();

            if let Some(drag_state) = drag.as_ref() {
                if drag_state.active && drag_state.mode == DragMode::ZoomRect {
                    if let (Some(rect), Some(transform)) =
                        (state.selection_rect.take(), state.transform.clone())
                    {
                        let rect = normalized_rect(rect);
                        if let Ok(mut plot) = self.plot.write() {
                            if let Some(viewport) = plot.viewport() {
                                if let Some(next) = zoom_to_rect(viewport, rect, &transform) {
                                    self.apply_manual_view_with_link(
                                        &mut plot,
                                        &mut state,
                                        transform.screen(),
                                        next,
                                    );
                                    self.publish_brush_link(Some(next.x));
                                    changed_view = Some(next);
                                }
                            }
                        }
                    }
                }
            }

            let click = state.pending_click.take();
            let should_toggle = click.as_ref().is_some_and(|click| {
                click.button == MouseButton::Left && click.region == HitRegion::Plot
            }) && drag.as_ref().is_none_or(|drag| !drag.active)
                && ev.click_count == 1;

            if should_toggle {
                if let Some(transform) = state.transform.clone() {
                    if let Some(data_point) = transform.screen_to_data(pos) {
                        state.selected_x = Some(data_point.x);
                        selected_x = Some(data_point.x);
                    }
                    if let Ok(mut plot) = self.plot.write() {
                        let target = state
                            .hover_target
                            .filter(|target| {
                                hover_target_within_threshold(target, pos, &self.config)
                            })
                            .or_else(|| {
                                compute_hover_target(
                                    &plot,
                                    &transform,
                                    pos,
                                    state.plot_rect,
                                    self.config.pin_threshold_px,
                                    self.config.unpin_threshold_px,
                                )
                            });

                        if let Some(target) = target {
                            let added = toggle_pin(plot.pins_mut(), target.pin);
                            let now = Instant::now();
                            state.last_pin_toggle = Some(PinToggle {
                                pin: target.pin,
                                added,
                                at: now,
                                screen_pos: target.screen,
                            });
                        }
                    }
                }
            } else if ev.click_count > 1 {
                state.last_pin_toggle = None;
            }

            state.drag = None;
            state.selection_rect = None;
            self.publish_cursor_link(None);
        }

        if let Some(x) = selected_x {
            cx.emit(PlotEvent::SelectedX(x));
        }
        if let Some(next) = changed_view {
            cx.emit(PlotEvent::ViewportChanged(Some(next)));
        }
        cx.notify();
    }

    fn on_mouse_up_out(&mut self, _ev: &MouseUpEvent, cx: &mut Context<Self>) {
        self.state
            .write()
            .expect("plot state lock")
            .clear_interaction();
        self.publish_cursor_link(None);
        cx.emit(PlotEvent::CursorX(None));
        cx.notify();
    }

    fn on_scroll(&mut self, ev: &ScrollEvent, _window: &Window, cx: &mut Context<Self>) {
        let pos = screen_point(ev.position);
        let mut changed_view = None;
        let mut handled = false;
        {
            let mut state = self.state.write().expect("plot state lock");
            if state.legend_hit(pos).is_some() {
                return;
            }
            let region = state.regions.hit_test(pos);
            let Some(transform) = state.transform.clone() else {
                return;
            };

            let line_height = px(16.0);
            let delta = ev.delta.pixel_delta(line_height);
            if ev.source == ScrollSource::Touch {
                // Mobile touch drags arrive as ScrollSource::Touch; pan instead of zooming.
                state.hover = None;
                let dx = f32::from(delta.x);
                let dy = f32::from(delta.y);
                if dx.abs() >= 0.01 || dy.abs() >= 0.01 {
                    if let Ok(mut plot) = self.plot.write()
                        && let Some(viewport) = plot.viewport()
                        && let Some(rect) = state.plot_rect
                    {
                        match region {
                            HitRegion::Plot => {
                                if let Some(next) =
                                    pan_viewport(viewport, ScreenPoint::new(dx, dy), &transform)
                                {
                                    self.apply_manual_view_with_link(
                                        &mut plot, &mut state, rect, next,
                                    );
                                    changed_view = Some(next);
                                }
                                handled = true;
                            }
                            HitRegion::XAxis => {
                                let factor = zoom_factor_from_drag(dx, rect.width().max(1.0));
                                let center = transform
                                    .screen_to_data(pos)
                                    .unwrap_or_else(|| viewport.x_center());
                                let next = zoom_viewport(viewport, center, factor, 1.0);
                                self.apply_manual_view_with_link(&mut plot, &mut state, rect, next);
                                changed_view = Some(next);
                                handled = true;
                            }
                            HitRegion::YAxis => {
                                let factor = zoom_factor_from_drag(-dy, rect.height().max(1.0));
                                let center = transform
                                    .screen_to_data(pos)
                                    .unwrap_or_else(|| viewport.y_center());
                                let next = zoom_viewport(viewport, center, 1.0, factor);
                                self.apply_manual_view_with_link(&mut plot, &mut state, rect, next);
                                changed_view = Some(next);
                                handled = true;
                            }
                            HitRegion::Outside => {}
                        }
                    }
                } else if region != HitRegion::Outside {
                    handled = true;
                }
            } else {
                let factor = scroll_zoom_factor(f32::from(delta.y));
                if factor == 1.0 {
                    return;
                }

                if let Ok(mut plot) = self.plot.write() {
                    if let Some(viewport) = plot.viewport() {
                        let center = transform
                            .screen_to_data(pos)
                            .unwrap_or_else(|| viewport.center());
                        let (factor_x, factor_y) = match region {
                            HitRegion::XAxis => (factor, 1.0),
                            HitRegion::YAxis => (1.0, factor),
                            HitRegion::Plot => (factor, factor),
                            HitRegion::Outside => (1.0, 1.0),
                        };
                        if factor_x != 1.0 || factor_y != 1.0 {
                            let next = zoom_viewport(viewport, center, factor_x, factor_y);
                            if let Some(rect) = state.plot_rect {
                                self.apply_manual_view_with_link(&mut plot, &mut state, rect, next);
                                changed_view = Some(next);
                                handled = true;
                            }
                        }
                    }
                }
            }
        }

        if handled {
            cx.stop_propagation();
        }
        if let Some(next) = changed_view {
            cx.emit(PlotEvent::ViewportChanged(Some(next)));
        }
        if handled || changed_view.is_some() {
            cx.notify();
        }
    }
}

impl Render for PlotView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let plot = Arc::clone(&self.plot);
        let state = Arc::clone(&self.state);
        let config = self.config.clone();
        let link = self.link.clone();
        let base_theme = plot.read().expect("plot lock").theme().clone();
        #[cfg(feature = "gpui_component_theme")]
        let theme = resolve_theme(base_theme, cx);
        #[cfg(not(feature = "gpui_component_theme"))]
        let theme = base_theme;
        let hover_region_id = Arc::as_ptr(&self.state) as usize;

        div()
            .id(("gpui-plot-view", hover_region_id))
            .size_full()
            .bg(to_hsla(theme.background))
            .child(
                canvas(
                    move |bounds, window, _app| {
                        let mut plot = plot.write().expect("plot lock");
                        let mut state = state.write().expect("plot state lock");
                        #[cfg(feature = "gpui_component_theme")]
                        if let Some(theme) = resolve_gpui_component_theme(_app) {
                            plot.set_theme(theme);
                        }
                        if let Some(link) = &link {
                            apply_link_updates(link, &mut plot, &mut state);
                        }
                        build_frame(&mut plot, &mut state, &config, bounds, window)
                    },
                    move |_, frame, window, cx| {
                        paint_frame(&frame, window, cx);
                    },
                )
                .size_full(),
            )
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, ev, _, cx| {
                    this.on_mouse_down(ev, cx);
                }),
            )
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(|this, ev, _, cx| {
                    this.on_mouse_down(ev, cx);
                }),
            )
            .on_mouse_move(cx.listener(|this, ev, _, cx| {
                this.on_mouse_move(ev, cx);
            }))
            .on_hover(cx.listener(|this, hovered, window, cx| {
                this.on_hover_state_change(*hovered, window, cx);
            }))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, ev, _, cx| {
                    this.on_mouse_up(ev, cx);
                }),
            )
            .on_mouse_up(
                MouseButton::Right,
                cx.listener(|this, ev, _, cx| {
                    this.on_mouse_up(ev, cx);
                }),
            )
            .on_mouse_up_out(
                MouseButton::Left,
                cx.listener(|this, ev, _, cx| {
                    this.on_mouse_up_out(ev, cx);
                }),
            )
            .on_mouse_up_out(
                MouseButton::Right,
                cx.listener(|this, ev, _, cx| {
                    this.on_mouse_up_out(ev, cx);
                }),
            )
            .on_scroll(cx.listener(|this, ev, window, cx| {
                this.on_scroll(ev, window, cx);
            }))
            .on_pinch(cx.listener(|this, ev: &PinchEvent, _, cx| {
                let factor = pinch_zoom_scale(ev.delta);
                if (factor - 1.0).abs() > f32::EPSILON {
                    this.zoom_by(factor, factor, Some(ev.position), cx);
                    cx.stop_propagation();
                }
            }))
    }
}

#[cfg(feature = "gpui_component_theme")]
fn resolve_gpui_component_theme(cx: &gpui_kit::App) -> Option<crate::style::Theme> {
    if cx.has_global::<gpui_kit::component::Theme>() {
        Some(crate::style::Theme::from_gpui_component_theme(
            gpui_kit::component::Theme::global(cx),
        ))
    } else {
        None
    }
}

#[cfg(feature = "gpui_component_theme")]
fn resolve_theme(base: crate::style::Theme, cx: &gpui_kit::App) -> crate::style::Theme {
    resolve_gpui_component_theme(cx).unwrap_or(base)
}

/// A handle for mutating a [`Plot`] held inside a [`PlotView`].
///
/// The handle clones cheaply and can be moved into async tasks.
#[derive(Debug, Clone)]
pub struct PlotHandle {
    plot: Arc<RwLock<Plot>>,
}

impl PlotHandle {
    /// Read the plot state.
    ///
    /// The plot is locked for the duration of the callback.
    pub fn read<R>(&self, f: impl FnOnce(&Plot) -> R) -> R {
        let plot = self.plot.read().expect("plot lock");
        f(&plot)
    }

    /// Mutate the plot state.
    ///
    /// The plot is locked for the duration of the callback.
    pub fn write<R>(&self, f: impl FnOnce(&mut Plot) -> R) -> R {
        let mut plot = self.plot.write().expect("plot lock");
        f(&mut plot)
    }
}

fn apply_link_updates(link: &LinkBinding, plot: &mut Plot, state: &mut PlotUiState) {
    if let Some(update) = link.group.latest_view_update()
        && update.seq > state.link_view_seq
    {
        state.link_view_seq = update.seq;
        if update.source != link.member_id {
            match update.kind {
                ViewSyncKind::Reset => {
                    if link.options.link_reset {
                        plot.reset_view();
                        state.viewport = None;
                        state.transform = None;
                        state.linked_brush_x = None;
                    }
                }
                ViewSyncKind::Manual {
                    viewport,
                    sync_x,
                    sync_y,
                } => {
                    let mut next = plot
                        .viewport()
                        .or_else(|| plot.data_bounds())
                        .unwrap_or(viewport);
                    let mut changed = false;
                    if sync_x && link.options.link_x {
                        next.x = viewport.x;
                        changed = true;
                    }
                    if sync_y && link.options.link_y {
                        next.y = viewport.y;
                        changed = true;
                    }
                    if changed {
                        plot.set_manual_view(next);
                        state.viewport = Some(next);
                        if let Some(rect) = state.plot_rect {
                            state.transform = Transform::new(next, rect);
                        }
                    }
                }
            }
        }
    }

    if let Some(update) = link.group.latest_cursor_update()
        && update.seq > state.link_cursor_seq
    {
        state.link_cursor_seq = update.seq;
        if update.source != link.member_id && link.options.link_cursor {
            state.linked_cursor_x = update.x;
        }
    }

    if let Some(update) = link.group.latest_brush_update()
        && update.seq > state.link_brush_seq
    {
        state.link_brush_seq = update.seq;
        if update.source != link.member_id && link.options.link_brush {
            state.linked_brush_x = update.x_range;
            if let Some(x_range) = update.x_range {
                let y_range = plot
                    .viewport()
                    .or_else(|| plot.data_bounds())
                    .map(|viewport| viewport.y)
                    .unwrap_or_else(|| Range::new(0.0, 1.0));
                let next = Viewport::new(x_range, y_range);
                plot.set_manual_view(next);
                state.viewport = Some(next);
                if let Some(rect) = state.plot_rect {
                    state.transform = Transform::new(next, rect);
                }
            }
        }
    }
}

fn screen_point(point: Point<Pixels>) -> ScreenPoint {
    ScreenPoint::new(f32::from(point.x), f32::from(point.y))
}

fn apply_manual_view(
    plot: &mut Plot,
    state: &mut PlotUiState,
    rect: ScreenRect,
    viewport: Viewport,
) {
    plot.set_manual_view(viewport);
    state.viewport = Some(viewport);
    state.transform = Transform::new(viewport, rect);
}

fn revert_pin_toggle(plot: &mut Plot, toggle: PinToggle) {
    let pins = plot.pins_mut();
    if toggle.added {
        if let Some(index) = pins.iter().position(|pin| *pin == toggle.pin) {
            pins.swap_remove(index);
        }
    } else if !pins.contains(&toggle.pin) {
        pins.push(toggle.pin);
    }
}

fn is_drag_button_held(mode: DragMode, pressed_button: Option<MouseButton>) -> bool {
    let expected = match mode {
        DragMode::ZoomRect => MouseButton::Right,
        DragMode::Pan | DragMode::ZoomX | DragMode::ZoomY => MouseButton::Left,
    };
    pressed_button == Some(expected)
}

fn scroll_zoom_factor(delta_y: f32) -> f64 {
    if delta_y.abs() < 0.01 {
        return 1.0;
    }

    (1.0 - (delta_y as f64 * 0.002)).clamp(0.1, 10.0)
}

fn pinch_zoom_scale(delta: f32) -> f32 {
    if !delta.is_finite() || delta.abs() < 0.0001 {
        return 1.0;
    }
    delta.exp().clamp(0.1, 10.0)
}

trait ViewportCenter {
    fn center(&self) -> DataPoint;
    fn x_center(&self) -> DataPoint;
    fn y_center(&self) -> DataPoint;
}

impl ViewportCenter for Viewport {
    fn center(&self) -> DataPoint {
        DataPoint::new(
            (self.x.min + self.x.max) * 0.5,
            (self.y.min + self.y.max) * 0.5,
        )
    }

    fn x_center(&self) -> DataPoint {
        DataPoint::new(
            (self.x.min + self.x.max) * 0.5,
            (self.y.min + self.y.max) * 0.5,
        )
    }

    fn y_center(&self) -> DataPoint {
        DataPoint::new(
            (self.x.min + self.x.max) * 0.5,
            (self.y.min + self.y.max) * 0.5,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DragMode, MouseButton, ScreenPoint, is_drag_button_held, pan_viewport, pinch_zoom_scale,
        scroll_zoom_factor,
    };
    use crate::geom::ScreenRect;
    use crate::transform::Transform;
    use crate::view::{Range, Viewport};

    #[test]
    fn drag_requires_matching_button() {
        assert!(is_drag_button_held(DragMode::Pan, Some(MouseButton::Left)));
        assert!(is_drag_button_held(
            DragMode::ZoomX,
            Some(MouseButton::Left)
        ));
        assert!(is_drag_button_held(
            DragMode::ZoomY,
            Some(MouseButton::Left)
        ));
        assert!(is_drag_button_held(
            DragMode::ZoomRect,
            Some(MouseButton::Right)
        ));
        assert!(!is_drag_button_held(
            DragMode::Pan,
            Some(MouseButton::Right)
        ));
        assert!(!is_drag_button_held(DragMode::ZoomRect, None));
    }

    #[test]
    fn positive_scroll_delta_zooms_in_after_reversal() {
        let factor = scroll_zoom_factor(120.0);
        assert!(factor < 1.0, "expected zoom-in factor, got {factor}");
    }

    #[test]
    fn negative_scroll_delta_zooms_out_after_reversal() {
        let factor = scroll_zoom_factor(-120.0);
        assert!(factor > 1.0, "expected zoom-out factor, got {factor}");
    }

    #[test]
    fn positive_pinch_delta_zooms_in_and_touch_pan_moves_opposite_drag() {
        assert!(pinch_zoom_scale(0.1) > 1.0);
        assert!(pinch_zoom_scale(-0.1) < 1.0);
        assert_eq!(pinch_zoom_scale(f32::NAN), 1.0);

        let viewport = Viewport::new(Range::new(100.0, 200.0), Range::new(0.0, 10.0));
        let rect = ScreenRect::new(ScreenPoint::new(0.0, 0.0), ScreenPoint::new(100.0, 100.0));
        let transform = Transform::new(viewport, rect).unwrap();
        let panned = pan_viewport(viewport, ScreenPoint::new(10.0, 0.0), &transform).unwrap();
        assert!(panned.x.min < viewport.x.min);
        assert_eq!(panned.y, viewport.y);
    }
}
