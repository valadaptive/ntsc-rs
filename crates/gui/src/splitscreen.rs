use eframe::{
    egui::{self, pos2, Context, Id, Rect, Sense, Widget},
    emath::{lerp, remap_clamp},
};

#[derive(Clone, Copy, Debug)]
enum DraggedEdge {
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Clone, Copy, Debug)]
struct DragState {
    dragged_edge: DraggedEdge,
    start_rect: Rect,
}

#[derive(Clone, Copy, Debug, Default)]
struct State {
    drag_state: Option<DragState>,
}

impl State {
    fn load(ctx: &Context, id: Id) -> Option<Self> {
        ctx.data_mut(|d| d.get_temp(id))
    }

    fn store(self, ctx: &Context, id: Id) {
        ctx.data_mut(|d| d.insert_temp(id, self));
    }
}

pub struct SplitScreen<'a> {
    value: &'a mut Rect,
}

impl<'a> SplitScreen<'a> {
    pub fn new(value: &'a mut Rect) -> Self {
        Self { value }
    }
}

impl<'a> Widget for SplitScreen<'a> {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let desired_size = ui.available_size();

        let grab_radius = ui.style().interaction.resize_grab_radius_side;
        let (id, rect) = ui.allocate_space(desired_size);

        let mut state = State::load(ui.ctx(), id).unwrap_or_default();

        let lerp_rect = Rect::from_min_max(
            pos2(
                lerp(rect.min.x..=rect.max.x, self.value.left()),
                lerp(rect.min.y..=rect.max.y, self.value.top()),
            ),
            pos2(
                lerp(rect.min.x..=rect.max.x, self.value.right()),
                lerp(rect.min.y..=rect.max.y, self.value.bottom()),
            ),
        );

        let mut response_left = {
            let interact_rect = Rect::from_min_max(
                egui::pos2(lerp_rect.left() - grab_radius, lerp_rect.top()),
                egui::pos2(lerp_rect.left() + grab_radius, lerp_rect.bottom()),
            );
            ui.interact(interact_rect, id.with("left"), Sense::drag())
        };

        let mut response_right = {
            let interact_rect = Rect::from_min_max(
                egui::pos2(lerp_rect.right() - grab_radius, lerp_rect.top()),
                egui::pos2(lerp_rect.right() + grab_radius, lerp_rect.bottom()),
            );
            ui.interact(interact_rect, id.with("right"), Sense::drag())
        };

        let mut response_top = {
            let interact_rect = Rect::from_min_max(
                egui::pos2(lerp_rect.left(), lerp_rect.top() - grab_radius),
                egui::pos2(lerp_rect.right(), lerp_rect.top() + grab_radius),
            );
            ui.interact(interact_rect, id.with("top"), Sense::drag())
        };

        let mut response_bottom = {
            let interact_rect = Rect::from_min_max(
                egui::pos2(lerp_rect.left(), lerp_rect.bottom() - grab_radius),
                egui::pos2(lerp_rect.right(), lerp_rect.bottom() + grab_radius),
            );
            ui.interact(interact_rect, id.with("bottom"), Sense::drag())
        };

        if response_left.hovered()
            || response_left.dragged()
            || response_right.hovered()
            || response_right.dragged()
        {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        } else if response_top.hovered()
            || response_top.dragged() | response_bottom.hovered()
            || response_bottom.dragged()
        {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
        }

        let mut set_dragged_edge = |edge| {
            if state.drag_state.is_none() {
                state.drag_state = Some(DragState {
                    dragged_edge: edge,
                    start_rect: *self.value,
                });
            }
        };

        if let (false, false, false, false) = (
            response_left.dragged(),
            response_right.dragged(),
            response_top.dragged(),
            response_bottom.dragged(),
        ) {
            state.drag_state = None;
        } else {
            response_left
                .dragged()
                .then(|| set_dragged_edge(DraggedEdge::Left));
            response_right
                .dragged()
                .then(|| set_dragged_edge(DraggedEdge::Right));
            response_top
                .dragged()
                .then(|| set_dragged_edge(DraggedEdge::Top));
            response_bottom
                .dragged()
                .then(|| set_dragged_edge(DraggedEdge::Bottom));

            match state.drag_state {
                Some(DragState {
                    dragged_edge: DraggedEdge::Right,
                    start_rect,
                }) => {
                    let pointer_position_2d = response_right.interact_pointer_pos().unwrap();
                    let position = pointer_position_2d.x;
                    let normalized = remap_clamp(position, rect.x_range(), 0.0..=1.0);
                    if normalized < start_rect.left() {
                        self.value.set_right(start_rect.left());
                        self.value.set_left(normalized);
                    } else {
                        self.value.set_right(normalized);
                        self.value.set_left(start_rect.left());
                    }
                    response_right.mark_changed();
                }
                Some(DragState {
                    dragged_edge: DraggedEdge::Left,
                    start_rect,
                }) => {
                    let pointer_position_2d = response_left.interact_pointer_pos().unwrap();
                    let position = pointer_position_2d.x;
                    let normalized = remap_clamp(position, rect.x_range(), 0.0..=1.0);
                    if normalized > start_rect.right() {
                        self.value.set_left(start_rect.right());
                        self.value.set_right(normalized);
                    } else {
                        self.value.set_left(normalized);
                        self.value.set_right(start_rect.right());
                    }
                    response_left.mark_changed();
                }
                Some(DragState {
                    dragged_edge: DraggedEdge::Top,
                    start_rect,
                }) => {
                    let pointer_position_2d = response_top.interact_pointer_pos().unwrap();
                    let position = pointer_position_2d.y;
                    let normalized = remap_clamp(position, rect.y_range(), 0.0..=1.0);
                    if normalized > start_rect.bottom() {
                        self.value.set_top(start_rect.bottom());
                        self.value.set_bottom(normalized);
                    } else {
                        self.value.set_top(normalized);
                        self.value.set_bottom(start_rect.bottom());
                    }
                    response_top.mark_changed();
                }
                Some(DragState {
                    dragged_edge: DraggedEdge::Bottom,
                    start_rect,
                }) => {
                    let pointer_position_2d = response_bottom.interact_pointer_pos().unwrap();
                    let position = pointer_position_2d.y;
                    let normalized = remap_clamp(position, rect.y_range(), 0.0..=1.0);
                    if normalized < start_rect.top() {
                        self.value.set_bottom(start_rect.top());
                        self.value.set_top(normalized);
                    } else {
                        self.value.set_bottom(normalized);
                        self.value.set_top(start_rect.top());
                    }
                    response_bottom.mark_changed();
                }
                None => {}
            }
        }

        state.store(ui.ctx(), id);

        if ui.is_rect_visible(rect) {
            let painter = ui.painter();

            // Left edge
            painter.line_segment(
                [
                    egui::pos2(lerp_rect.left(), lerp_rect.top()),
                    egui::pos2(lerp_rect.left(), lerp_rect.bottom()),
                ],
                ui.style().interact(&response_left).fg_stroke,
            );
            // Right edge
            painter.line_segment(
                [
                    egui::pos2(lerp_rect.right(), lerp_rect.top()),
                    egui::pos2(lerp_rect.right(), lerp_rect.bottom()),
                ],
                ui.style().interact(&response_right).fg_stroke,
            );
            // Top edge
            painter.line_segment(
                [
                    egui::pos2(lerp_rect.left(), lerp_rect.top()),
                    egui::pos2(lerp_rect.right(), lerp_rect.top()),
                ],
                ui.style().interact(&response_top).fg_stroke,
            );
            // Bottom edge
            painter.line_segment(
                [
                    egui::pos2(lerp_rect.left(), lerp_rect.bottom()),
                    egui::pos2(lerp_rect.right(), lerp_rect.bottom()),
                ],
                ui.style().interact(&response_bottom).fg_stroke,
            );

            // Fill with red rectangle in debug mode
            #[cfg(debug_assertions)]
            if ui.ctx().debug_on_hover() && ui.interact(rect, id, Sense::hover()).hovered() {
                painter.rect_filled(
                    rect,
                    egui::Rounding::ZERO,
                    egui::Color32::from_rgba_unmultiplied(255, 0, 0, 64),
                );
            }
        }

        response_right
            .union(response_left)
            .union(response_top)
            .union(response_bottom)
    }
}
