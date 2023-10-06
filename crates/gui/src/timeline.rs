use std::ops::RangeInclusive;
use std::f32;

use eframe::{
    egui::{self, vec2, Context, Id, Sense, TextStyle, Widget},
    emath::{remap, remap_clamp},
    epaint::{pos2, Color32, Pos2, Stroke},
};
use gstreamer::{ClockTime, Fraction};

use crate::gst_utils::clock_format::clock_time_format;

type GetSetValue<'a> = Box<dyn 'a + FnMut(Option<u64>) -> u64>;

fn get(get_set_value: &mut GetSetValue<'_>) -> u64 {
    (get_set_value)(None)
}

fn set(get_set_value: &mut GetSetValue<'_>, value: u64) {
    (get_set_value)(Some(value));
}

fn wrap_modulo(n: f64, modulus: f64) -> f64 {
    ((n % modulus) + modulus) % modulus
}

#[derive(Clone, Copy, Debug)]
struct State {
    // RangeInclusive isn't Copy
    zoom_range: [f64; 2],
}

impl Default for State {
    fn default() -> Self {
        Self {
            zoom_range: [0.0, 1.0],
        }
    }
}

impl State {
    fn load(ctx: &Context, id: Id) -> Option<Self> {
        ctx.data_mut(|d| d.get_temp(id))
    }

    fn store(self, ctx: &Context, id: Id) {
        ctx.data_mut(|d| d.insert_temp(id, self));
    }
}

pub struct Timeline<'a> {
    get_set_value: GetSetValue<'a>,
    range: RangeInclusive<u64>,
    framerate: Option<Fraction>,
}

const THICKNESS: f32 = 24.0;

impl<'a> Timeline<'a> {
    pub fn new(
        value: &'a mut u64,
        range: RangeInclusive<u64>,
        framerate: Option<Fraction>,
    ) -> Self {
        Self {
            range,
            get_set_value: Box::new(move |v: Option<u64>| {
                if let Some(v) = v {
                    *value = v
                }
                *value
            }),
            framerate,
        }
    }
}

// egui has trouble tessellating concave shapes, so we do it manually, feathering and all
fn make_cursor_shape(
    bottom_center: Pos2,
    height: f32,
    thickness: f32,
    triangle_radius: f32,
    color: egui::Color32,
    feather_diameter: f32,
) -> egui::Shape {
    let feather_radius = feather_diameter * 0.5;
    let mut vertices: Vec<egui::epaint::Vertex> = Vec::new();
    let mut add_vertex = |x: f32, y: f32, feather: bool| {
        vertices.push(egui::epaint::Vertex {
            pos: pos2(x, y),
            uv: egui::epaint::WHITE_UV,
            color: if feather {Color32::TRANSPARENT} else {color},
        });
        (vertices.len() - 1) as u32
    };

    let angle_y_scale: f32 = (std::f32::consts::PI / 8.0).tan();

    let top_left = add_vertex(
        bottom_center.x - (thickness * 0.5) + feather_radius,
        bottom_center.y - height,
        false
    );
    let top_left_feather = add_vertex(
        bottom_center.x - (thickness * 0.5) - feather_radius,
        bottom_center.y - height,
        true
    );
    let top_right = add_vertex(
        bottom_center.x + (thickness * 0.5) - feather_radius,
        bottom_center.y - height,
        false
    );
    let top_right_feather = add_vertex(
        bottom_center.x + (thickness * 0.5) + feather_radius,
        bottom_center.y - height,
        true
    );
    let bottom_left_rect = add_vertex(
        bottom_center.x - (thickness * 0.5) + feather_radius,
        bottom_center.y - triangle_radius + (feather_radius * angle_y_scale),
        false
    );
    let bottom_left_rect_feather = add_vertex(
        bottom_center.x - (thickness * 0.5) - feather_radius,
        bottom_center.y - triangle_radius - (feather_radius * angle_y_scale),
        true
    );
    let bottom_right_rect = add_vertex(
        bottom_center.x + (thickness * 0.5) - feather_radius,
        bottom_center.y - triangle_radius + (feather_radius * angle_y_scale),
        false
    );
    let bottom_right_rect_feather = add_vertex(
        bottom_center.x + (thickness * 0.5) + feather_radius,
        bottom_center.y - triangle_radius - (feather_radius * angle_y_scale),
        true
    );
    let bottom_left_tri = add_vertex(
        bottom_center.x - (thickness * 0.5) - triangle_radius + (feather_radius * (1.0 + angle_y_scale)),
        bottom_center.y,
        false
    );
    let bottom_left_tri_feather = add_vertex(
        bottom_center.x - (thickness * 0.5) - triangle_radius - (feather_radius * (1.0 + angle_y_scale)),
        bottom_center.y,
        true
    );
    let bottom_right_tri = add_vertex(
        bottom_center.x + (thickness * 0.5) + triangle_radius - (feather_radius * (1.0 + angle_y_scale)),
        bottom_center.y,
        false
    );
    let bottom_right_tri_feather = add_vertex(
        bottom_center.x + (thickness * 0.5) + triangle_radius + (feather_radius * (1.0 + angle_y_scale)),
        bottom_center.y,
        true
    );

    #[rustfmt::skip]
    let indices = vec![
        top_left, top_right, bottom_left_rect,
        top_right, bottom_left_rect, bottom_right_rect,
        bottom_left_rect, bottom_left_tri, bottom_right_tri,
        bottom_left_rect, bottom_right_rect, bottom_right_tri,
        top_left_feather, top_left, bottom_left_rect,
        top_left_feather, bottom_left_rect_feather, bottom_left_rect,
        bottom_left_rect_feather, bottom_left_rect, bottom_left_tri,
        bottom_left_tri_feather,bottom_left_tri, bottom_left_rect_feather,
        top_right_feather, top_right, bottom_right_rect,
        top_right_feather, bottom_right_rect_feather, bottom_right_rect,
        bottom_right_rect_feather, bottom_right_rect, bottom_right_tri,
        bottom_right_tri_feather,bottom_right_tri, bottom_right_rect_feather,
    ];

    egui::Shape::Mesh(egui::epaint::Mesh {
        indices,
        vertices,
        texture_id: egui::TextureId::Managed(0),
    })
}

impl<'a> Widget for Timeline<'a> {
    fn ui(mut self, ui: &mut egui::Ui) -> egui::Response {
        let old_value = get(&mut self.get_set_value);

        let desired_size = vec2(
            ui.available_width(),
            THICKNESS.max(ui.spacing().interact_size.y),
        );

        let mut response = ui.allocate_response(desired_size, Sense::drag());

        let mut state = State::load(ui.ctx(), response.id).unwrap_or_default();

        let rect = response.rect;
        let position_range = rect.x_range();
        let position_range_f64 = position_range.min as f64..=position_range.max as f64;
        let range_f64 = *self.range.start() as f64..=*self.range.end() as f64;

        if ui.rect_contains_pointer(rect) {
            let scroll_delta = ui.ctx().input(|input| input.scroll_delta);
            let zoom_delta = ui.ctx().input(|input| input.zoom_delta());
            let scroll_delta = scroll_delta.x + scroll_delta.y;

            if zoom_delta != 1.0 {
                let pointer_pos = ui.ctx().input(|i| i.pointer.hover_pos());

                if let Some(pointer_pos) = pointer_pos {
                    let scaled_position = ((pointer_pos.x - rect.left()) / rect.width()) as f64;
                    let remapped_position = remap_clamp(
                        scaled_position,
                        0.0..=1.0,
                        state.zoom_range[0]..=state.zoom_range[1],
                    );

                    state.zoom_range[0] = ((state.zoom_range[0] - remapped_position)
                        / zoom_delta as f64
                        + remapped_position)
                        .clamp(0.0, 1.0);
                    state.zoom_range[1] = ((state.zoom_range[1] - remapped_position)
                        / zoom_delta as f64
                        + remapped_position)
                        .clamp(0.0, 1.0);

                    const MAX_ZOOM_RANGE: f64 = 0.05;
                    if state.zoom_range[1] - state.zoom_range[0] < MAX_ZOOM_RANGE {
                        state.zoom_range[0] =
                            remapped_position - (MAX_ZOOM_RANGE * remapped_position);
                        state.zoom_range[1] =
                            remapped_position + (MAX_ZOOM_RANGE * (1.0 - remapped_position));
                        if state.zoom_range[0] < 0.0 {
                            state.zoom_range[0] = 0.0;
                            state.zoom_range[1] = MAX_ZOOM_RANGE;
                        }
                        if state.zoom_range[1] > 1.0 {
                            state.zoom_range[1] = 1.0;
                            state.zoom_range[0] = 1.0 - MAX_ZOOM_RANGE;
                        }
                    }
                }
            }

            if scroll_delta != 0.0 {
                dbg!(scroll_delta);
                let zoom_span = state.zoom_range[1] - state.zoom_range[0];
                // we need to negate the scroll delta--scrolling down and right are both negative?
                let delta = (-scroll_delta.signum() / rect.width()) as f64 * zoom_span;
                if delta > 0.0 {
                    state.zoom_range[1] = (state.zoom_range[1] + delta).min(1.0);
                    state.zoom_range[0] = state.zoom_range[1] - zoom_span;
                } else {
                    state.zoom_range[0] = (state.zoom_range[0] + delta).max(0.0);
                    state.zoom_range[1] = state.zoom_range[0] + zoom_span;
                }
            }

            state.store(ui.ctx(), response.id);
        }

        let scaled_range = (range_f64.start()
            + ((range_f64.end() - range_f64.start()) * state.zoom_range[0]))
            ..=((range_f64.end() - range_f64.start()) * state.zoom_range[1] + range_f64.start());

        if let Some(pointer_position_2d) = response.interact_pointer_pos() {
            let position = pointer_position_2d.x;
            let normalized = remap_clamp(
                position as f64,
                position_range_f64.clone(),
                scaled_range.clone(),
            ) as u64;
            set(&mut self.get_set_value, normalized);
        }

        if ui.is_rect_visible(rect) {
            let value = get(&mut self.get_set_value);
            let x = remap(
                value as f64,
                scaled_range.clone(),
                position_range_f64.clone(),
            ) as f32;
            let widget_visuals = ui.visuals();
            let visuals = ui.style().interact(&response);

            let painter = ui.painter().with_clip_rect(rect);
            let feathering = ui.ctx().tessellation_options(|t| t.feathering_size_in_pixels) / ui.ctx().pixels_per_point();

            // Draw background
            painter.rect_filled(rect, egui::Rounding::ZERO, visuals.bg_fill);

            let fine_tick_mark_interval = if let Some(framerate) = self.framerate {
                framerate.denom() as f64 * ClockTime::SECOND.nseconds() as f64
                    / framerate.numer() as f64
            } else {
                0.25 * ClockTime::SECOND.nseconds() as f64
            };

            // Find the finest tick mark interval that doesn't result in the tick marks being spaced <4px apart.
            // If none exists, don't render tick marks.
            let tick_mark_intervals = [
                fine_tick_mark_interval,
                1.0 * ClockTime::SECOND.nseconds() as f64,
                60.0 * ClockTime::SECOND.nseconds() as f64,
            ];
            let mut i: usize = 0;
            let tick_mark_info = loop {
                let tick_mark_interval = tick_mark_intervals.get(i);
                if let Some(tick_mark_interval) = tick_mark_interval {
                    let num_tick_marks = ((scaled_range.end() - scaled_range.start())
                        / tick_mark_interval)
                        .ceil() as usize;
                    if num_tick_marks as f32 <= rect.width() / 4.0 {
                        break Some((num_tick_marks, tick_mark_intervals[i]));
                    }
                } else {
                    break None;
                }
                i += 1;
            };

            if let Some((num_tick_marks, tick_mark_interval)) = tick_mark_info {
                let range_to_screen = (position_range_f64.end() - position_range_f64.start())
                    / (scaled_range.end() - scaled_range.start());
                // Euclidean modulo (wrap even when negative)
                let tick_mark_start = (tick_mark_interval
                    - wrap_modulo(*scaled_range.start(), tick_mark_interval))
                    * range_to_screen;
                // draw an extra tick mark because even if the first one is negative, the stroke has a nonzero width and we
                // may need to draw it anyway
                for i in 0..=num_tick_marks {
                    let tick_mark_pos =
                        tick_mark_start + ((i as f64 - 1.0) * tick_mark_interval * range_to_screen);
                    painter.line_segment(
                        [
                            pos2(tick_mark_pos as f32 + rect.left(), rect.bottom()),
                            pos2(
                                tick_mark_pos as f32 + rect.left(),
                                rect.bottom() - (rect.height() / 3.0),
                            ),
                        ],
                        Stroke {
                            width: 1.0,
                            color: widget_visuals.text_color(),
                        },
                    );
                }
            }

            const TRIANGLE_RADIUS: f32 = 4.0;

            if x >= -TRIANGLE_RADIUS {
                // Draw time cursor
                painter.add(make_cursor_shape(
                    pos2(x, rect.bottom()),
                    rect.height(),
                    2.0,
                    TRIANGLE_RADIUS,
                    visuals.fg_stroke.color,
                    feathering
                ));
            }

            // Draw start and end time
            painter.text(
                rect.left_top(),
                egui::Align2::LEFT_TOP,
                clock_time_format(*scaled_range.start() as u64),
                TextStyle::Body.resolve(ui.style()),
                visuals.text_color(),
            );
            painter.text(
                rect.right_top(),
                egui::Align2::RIGHT_TOP,
                clock_time_format(*scaled_range.end() as u64),
                TextStyle::Body.resolve(ui.style()),
                visuals.text_color(),
            );
        }

        response.changed = get(&mut self.get_set_value) != old_value;

        response
    }
}
