#include "ntscrs-ae.h"
#include <stdio.h>
#include <stdlib.h>

static ntscrs_pixel_format ntscrs_world_pixel_format_to_pixel_format (PF_PixelFormat in_pix_fmt) {
	switch (in_pix_fmt) {
		case PF_PixelFormat_ARGB32: return NTSCRS_XRGB8;
		case PF_PixelFormat_ARGB64: return NTSCRS_XRGB16AE;
		case PF_PixelFormat_ARGB128: return NTSCRS_XRGB32F;
		default: return NTSCRS_INVALID_PIXEL_FORMAT;
	}
}


static ntscrs_pixel_format ntscrs_premiere_pixel_format_to_pixel_format (PF_PixelFormat in_pix_fmt) {
	switch (in_pix_fmt) {
		case PrPixelFormat_BGRA_4444_8u: return NTSCRS_BGRX8;
		case PrPixelFormat_BGRA_4444_16u: return NTSCRS_BGRX16;
		case PrPixelFormat_BGRA_4444_32f: return NTSCRS_BGRX32F;
		default: return NTSCRS_INVALID_PIXEL_FORMAT;
	}
}

static A_long ceil_div(A_long a, A_long b) {
	return (a / b) + (a % b != 0);
}

static PF_Err
About (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	PF_SPRINTF(	out_data->return_msg,
				"%s %d.%d.%d\r\r%s",
				StrID_Name,
				MAJOR_VERSION,
				MINOR_VERSION,
				BUG_VERSION,
				StrID_Description);

	return PF_Err_NONE;
}

static PF_Err
GlobalSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	out_data->my_version 	= 	PF_VERSION(	MAJOR_VERSION,
											MINOR_VERSION,
											BUG_VERSION,
											STAGE_VERSION,
											BUILD_VERSION);

	out_data->out_flags |=	PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_SEND_UPDATE_PARAMS_UI | PF_OutFlag_NON_PARAM_VARY;

	out_data->out_flags2 	=	PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG	|
								PF_OutFlag2_SUPPORTS_SMART_RENDER				|
								PF_OutFlag2_FLOAT_COLOR_AWARE					|
								PF_OutFlag2_REVEALS_ZERO_ALPHA					|
								PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

	AEGP_SuiteHandler suites(in_data->pica_basicP);
	// It's fine to store our Rust plugin data in a Handle (which can move around) because it's allocated on the Rust
	// side and only a pointer to it is given to us. That allocation won't move even if the Handle does
	PF_Handle data_handle = suites.HandleSuite1()->host_new_handle(sizeof(NtscAE_GlobalData));
	if (!data_handle) {
		return PF_Err_OUT_OF_MEMORY;
	}
	out_data->global_data = data_handle;

	if (in_data->appl_id == 'PrMr') {
		AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite =
			AEFX_SuiteScoper<PF_PixelFormatSuite1>(	in_data,
													kPFPixelFormatSuite,
													kPFPixelFormatSuiteVersion1,
													out_data);
		(*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);
		// I *could* add support for BGRX, RGBA, RGBX, etc but I have no way to test those code paths
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref,
													 PrPixelFormat_BGRA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref,
													 PrPixelFormat_BGRA_4444_16u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref,
													 PrPixelFormat_BGRA_4444_8u);
	}

	return PF_Err_NONE;
}

static PF_Err
PreRender(
	PF_InData				*in_data,
	PF_OutData				*out_data,
	PF_PreRenderExtra		*extra)
{
	PF_Err err = PF_Err_NONE;
	PF_ParamDef channel_param;
	PF_RenderRequest req = extra->input->output_request;
	PF_CheckoutResult in_result;

	req.preserve_rgb_of_zero_alpha = TRUE;

	// I don't know anymore. I thought the way to ask for really all the pixels (yes I promise I need all of them for
	// the effect to work!) was to set result_rect with the helpfully-provided ref_width and ref_height properties from
	// the checkout result. Apparently, that's too *late*, and we need to tell the request itself before we check out
	// the layer that we're planning on checking out these pixels. No thanks to the SmartFX documentation, which fails
	// to mention which parameters have an effect on the input pixel region we're allowed to access, which ones we're
	// even *allowed to modify* (it mixes input and output parameters in the same structs!), and half the time, what
	// coordinate space we're even working in and whether downsampling is taken into account. I'm 70% sure this is
	// *still* wrong somehow, but it seems to give the correct results. I hope whoever designed the SmartFX API and
	// *especially* whoever wrote the paltry and inadequate documentation for it are cursed with wet socks for eternity.
	req.rect.left = 0;
	req.rect.right = ceil_div(in_data->width * in_data->downsample_x.num, in_data->downsample_x.den);
	req.rect.top = 0;
	req.rect.bottom = ceil_div(in_data->height * in_data->downsample_y.num, in_data->downsample_y.den);

	ERR(extra->cb->checkout_layer(	in_data->effect_ref,
									LAYER_INPUT,
									LAYER_INPUT,
									&req,
									in_data->current_time,
									in_data->time_step,
									in_data->time_scale,
									&in_result));

	// Completely ignore previous bounds, and just use the layer bounds (for adjustment layers and shape layers, these
	// are the comp bounds). This was surprisingly tricky to figure out--you'd think lots of effects would want to do
	// this.
	A_long out_w = ceil_div(in_result.ref_width * in_data->downsample_x.num, in_data->downsample_x.den);
	A_long out_h = ceil_div(in_result.ref_height * in_data->downsample_y.num, in_data->downsample_y.den);

	PF_Point out_origin = {
		(A_short)in_data->pre_effect_source_origin_x,
		(A_short)in_data->pre_effect_source_origin_y
	};

	PF_Rect constrained_rect = {
		0,
		0,
		out_w,
		out_h,
	};

	extra->output->result_rect = extra->output->max_result_rect = constrained_rect;
	extra->output->flags |= PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS;

	// The SDK example says:
	// "For SmartFX, AE automagically checks in any params checked out
	// during PF_Cmd_SMART_PRE_RENDER, new or old-fashioned."
	// I sure hope they're not lying!

	return err;
}

static inline float clamp(float value, float min, float max) {
	return value < min ? min : value > max ? max : value;
}

#define LOG_SLIDER_BASE 100.0

// Maps a slider value to a roughly logarithmic curve, keeping the min and max values where they are.
static float map_logarithmic(float value, float min, float max, float base) {
	return (max - min) * ((powf(base, (value - min) / (max - min)) - 1.0) / (base - 1.0)) + min;
}

// Map it back. This is used for converting default slider values.
static float map_logarithmic_inverse(float value, float min, float max, float base) {
	return (logf(((value - min) / (max - min)) * (base - 1) + 1) / logf(base)) * (max - min) + min;
}

// Check out effect params and map them to the given ntscrs Configurator
static PF_Err apply_ntsc_settings(
	PF_InData* in_data,
	PF_OutData* out_data,
	uint32_t* use_field_out,
	ntscrs_Configurator* configurator
) {
	PF_Err err = PF_Err_NONE;
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	NtscAE_GlobalData* global_data = reinterpret_cast<NtscAE_GlobalData*>(suites.HandleSuite1()->host_lock_handle(out_data->global_data));
	if (!global_data) {
		return PF_Err_BAD_CALLBACK_PARAM;
	}

	PF_ParamDef param_def;
	for (int i = 0; i < global_data->num_checkoutable_params; i++) {
		AEFX_CLR_STRUCT(param_def);

		NtscAE_CheckoutableParam p = global_data->checkoutable_params[i];
		ERR(PF_CHECKOUT_PARAM(in_data, p.index, in_data->current_time, in_data->time_step, in_data->time_scale, &param_def));
		if (err) break;

		const ntscrs_SettingDescriptor* descriptor = ntscrs_settingslist_get_descriptor_by_id(&global_data->settings, p.id);
		switch (descriptor->kind.tag) {
			case ntscrs_SettingKind_Enumeration: {
				// Not sure what item 0 maps to, but the menu items are indexed starting at 1
				A_long selected_item_position = param_def.u.pd.value - 1;
				if (selected_item_position > param_def.u.pd.value) continue;
				// Map menu item index back to enum value
				uint32_t menu_enum_value = descriptor->kind.enumeration.options[selected_item_position].index;
				// We need to handle the "use field" setting ourselves
				if (p.id == NTSCRS_USE_FIELD) {
					*use_field_out = menu_enum_value;
				} else {
					ntscrs_settings_set_field_int(configurator, p.id, menu_enum_value);
				}
				break;
			}
			case ntscrs_SettingKind_Percentage: {
				float slider_value = param_def.u.fs_d.value * 0.01;
				if (descriptor->kind.percentage.logarithmic) {
					slider_value = map_logarithmic(slider_value, 0.0, 1.0, LOG_SLIDER_BASE);
				}
				ntscrs_settings_set_field_float(configurator, p.id, slider_value);
				break;
			}
			case ntscrs_SettingKind_IntRange: {
				// AE only supports float sliders. When you tell it to display 0 decimals, it seems to round and not
				// floor.
				ntscrs_settings_set_field_int(configurator, p.id, (int32_t)(param_def.u.fs_d.value + 0.5));
				break;
			}
			case ntscrs_SettingKind_FloatRange: {
				float slider_value = param_def.u.fs_d.value;
				ntscrs_SettingKind_ntscrs_FloatRange_Body float_range = descriptor->kind.float_range;
				if (float_range.logarithmic) {
					slider_value = map_logarithmic(slider_value, float_range.min, float_range.max, LOG_SLIDER_BASE);
				}
				ntscrs_settings_set_field_float(configurator, p.id, param_def.u.fs_d.value);
				break;
			}
			case ntscrs_SettingKind_Boolean:
			// Groups map to a checkbox that controls whether they're enabled
			case ntscrs_SettingKind_Group: {
				ntscrs_settings_set_field_bool(configurator, p.id, param_def.u.bd.value != 0);
				break;
			}
		}
		ERR(PF_CHECKIN_PARAM(in_data, &param_def));
	}

	suites.HandleSuite1()->host_unlock_handle(out_data->global_data);
	return err;
}

static PF_Err
ActuallyRender(
	PF_InData		*in_data,
	PF_EffectWorld 	*input,
	PF_OutData		*out_data,
	PF_EffectWorld	*output,
	ntscrs_pixel_format input_pixel_format,
	ntscrs_pixel_format output_pixel_format)
{
	PF_Err				err 	= PF_Err_NONE,
						err2 	= PF_Err_NONE;
	PF_WorldSuite2		*wsP	=	NULL;

	AEGP_SuiteHandler suites(in_data->pica_basicP);

	if (input_pixel_format == NTSCRS_INVALID_PIXEL_FORMAT || output_pixel_format == NTSCRS_INVALID_PIXEL_FORMAT) {
		return PF_Err_BAD_CALLBACK_PARAM;
	}

	ERR(AEFX_AcquireSuite(	in_data,
							out_data,
							kPFWorldSuite,
							kPFWorldSuiteVersion2,
							StrID_Err_LoadSuite,
							(void**)&wsP));

	uint32_t use_field = 0;
	ntscrs_Configurator* configurator = ntscrs_configurator_create();
	apply_ntsc_settings(in_data, out_data, &use_field, configurator);

	if (!input) return err;

	if (!err) {
		// Number of YIQ rows/fields for ntsc-rs to operate on.
		A_long num_rows;
		A_long frame_num = in_data->current_time / in_data->time_step;

		ntscrs_YiqField yiq_field;
		switch (use_field) {
			case NTSCRS_USE_FIELD_ALTERNATING:
				yiq_field = frame_num & 1 ? ntscrs_YiqField_Upper : ntscrs_YiqField_Lower;
				break;
			case NTSCRS_USE_FIELD_UPPER:
				yiq_field = ntscrs_YiqField_Upper;
				break;
			case NTSCRS_USE_FIELD_LOWER:
				yiq_field = ntscrs_YiqField_Lower;
				break;
			case NTSCRS_USE_FIELD_BOTH:
				yiq_field = ntscrs_YiqField_Both;
				break;
			case NTSCRS_USE_FIELD_INTERLEAVED_UPPER:
				yiq_field = ntscrs_YiqField_InterleavedUpper;
				break;
			case NTSCRS_USE_FIELD_INTERLEAVED_LOWER:
				yiq_field = ntscrs_YiqField_InterleavedLower;
				break;
			default:
				return PF_Err_UNRECOGNIZED_PARAM_TYPE;
		}

		switch (yiq_field) {
			// On an image with an odd input height, we do ceiling division if we render upper-field-first (take an
			// image 3 pixels tall. it goes render, skip, render--that's 2 renders) but floor division if we render
			// lower-field-first (skip, render, skip--only 1 render).
			case ntscrs_YiqField_Upper:
				num_rows = (output->height + 1) / 2;
				break;
			case ntscrs_YiqField_Lower:
				num_rows = MAX(output->height / 2, 1);
				break;
			case ntscrs_YiqField_Both:
			case ntscrs_YiqField_InterleavedUpper:
			case ntscrs_YiqField_InterleavedLower:
				num_rows = output->height;
				break;
			default:
				return PF_Err_UNRECOGNIZED_PARAM_TYPE;
		}

		A_long output_plane_size = output->width * num_rows;
		// Allocate an intermediate buffer which the ntscrs library will operate on.
		size_t out_buf_size = sizeof(float) * 3 * output_plane_size;

		PF_Handle out_handle = suites.HandleSuite1()->host_new_handle(out_buf_size);
		if (out_handle) {
			float* out_buf = reinterpret_cast<float*>(suites.HandleSuite1()->host_lock_handle(out_handle));
			if (out_buf) {
				for (int i = 0; i < output_plane_size * 3; i++) {
					out_buf[i] = 0.0;
				}

				struct ntscrs_BlitInfo src_blit_info = {0};
				src_blit_info.row_bytes = input->rowbytes;
				src_blit_info.other_buffer_height = input->height;
				src_blit_info.destination_x = MAX(0, input->origin_x);
				src_blit_info.destination_y = MAX(0, input->origin_y);
				src_blit_info.rect.left = 0;
				src_blit_info.rect.top = 0;
				src_blit_info.rect.right = MIN(input->width, output->width - src_blit_info.destination_x);
				src_blit_info.rect.bottom = MIN(input->height, output->height - src_blit_info.destination_y);
				src_blit_info.flip_y = 0;

				struct ntscrs_BlitInfo dst_blit_info = {0};
				dst_blit_info.other_buffer_height = output->height;
				dst_blit_info.rect.left = 0;
				dst_blit_info.rect.top = 0;
				dst_blit_info.rect.right = output->width;
				dst_blit_info.rect.bottom = output->height;
				dst_blit_info.row_bytes = output->rowbytes;
				dst_blit_info.flip_y = 0;

				switch (input_pixel_format) {
					case NTSCRS_XRGB32F: {
						ntscrs_yiq_set_from_strided_buffer_Xrgb32f(
							(float_t*)input->data,
							out_buf,
							src_blit_info,
							output->width,
							output->height,
							yiq_field
						);
						break;
					}
					case NTSCRS_XRGB16AE: {
						ntscrs_yiq_set_from_strided_buffer_Xrgb16AE(
							(int16_t*)input->data,
							out_buf,
							src_blit_info,
							output->width,
							output->height,
							yiq_field
						);
						break;
					}
					case NTSCRS_XRGB8: {
						ntscrs_yiq_set_from_strided_buffer_Xrgb8(
							(uint8_t*)input->data,
							out_buf,
							src_blit_info,
							output->width,
							output->height,
							yiq_field
						);
						break;
					}
					case NTSCRS_BGRX32F: {
						ntscrs_yiq_set_from_strided_buffer_Bgrx32f(
							(float_t*)input->data,
							out_buf,
							src_blit_info,
							output->width,
							output->height,
							yiq_field
						);
						break;
					}
					case NTSCRS_BGRX16: {
						ntscrs_yiq_set_from_strided_buffer_Bgrx16(
							(uint16_t*)input->data,
							out_buf,
							src_blit_info,
							output->width,
							output->height,
							yiq_field
						);
						break;
					}
					case NTSCRS_BGRX8: {
						ntscrs_yiq_set_from_strided_buffer_Bgrx8(
							(uint8_t*)input->data,
							out_buf,
							src_blit_info,
							output->width,
							output->height,
							yiq_field
						);
						break;
					}
				}

				ntscrs_process_yiq(
					out_buf, // y plane
					&out_buf[output_plane_size], // i plane
					&out_buf[output_plane_size * 2], // q plane
					num_rows, output->width, output->height, configurator, frame_num, yiq_field);

				switch (output_pixel_format) {
					case NTSCRS_XRGB32F: {
						ntscrs_yiq_write_to_strided_buffer_Xrgb32f(
							out_buf,
							(float_t*)output->data,
							dst_blit_info,
							output->width,
							output->height,
							yiq_field
						);
						break;
					}
					case NTSCRS_XRGB16AE: {
						ntscrs_yiq_write_to_strided_buffer_Xrgb16AE(
							out_buf,
							(int16_t*)output->data,
							dst_blit_info,
							output->width,
							output->height,
							yiq_field
						);
						break;
					}
					case NTSCRS_XRGB8: {
						ntscrs_yiq_write_to_strided_buffer_Xrgb8(
							out_buf,
							(uint8_t*)output->data,
							dst_blit_info,
							output->width,
							output->height,
							yiq_field
						);
						break;
					}
					case NTSCRS_BGRX32F: {
						ntscrs_yiq_write_to_strided_buffer_Bgrx32f(
							out_buf,
							(float_t*)output->data,
							dst_blit_info,
							output->width,
							output->height,
							yiq_field
						);
						break;
					}
					case NTSCRS_BGRX16: {
						ntscrs_yiq_write_to_strided_buffer_Bgrx16(
							out_buf,
							(uint16_t*)output->data,
							dst_blit_info,
							output->width,
							output->height,
							yiq_field
						);
						break;
					}
					case NTSCRS_BGRX8: {
						ntscrs_yiq_write_to_strided_buffer_Bgrx8(
							out_buf,
							(uint8_t*)output->data,
							dst_blit_info,
							output->width,
							output->height,
							yiq_field
						);
						break;
					}
				}
				suites.HandleSuite1()->host_unlock_handle(out_handle);
			} else {
				err = PF_Err_INTERNAL_STRUCT_DAMAGED;
			}
			suites.HandleSuite1()->host_dispose_handle(out_handle);
		} else {
			err = PF_Err_INTERNAL_STRUCT_DAMAGED;
		}
	}

	ERR2(AEFX_ReleaseSuite(	in_data,
							out_data,
							kPFWorldSuite,
							kPFWorldSuiteVersion2,
							StrID_Err_FreeSuite));

	return err;
}

static PF_Err
Render (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	if (in_data->appl_id == 'PrMr') {
		PF_LayerDef* input = &params[LAYER_INPUT]->u.ld;

		if (input->width != output->width || input->height != output->height) {
			return PF_Err_BAD_CALLBACK_PARAM;
		}

		AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite = AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data,
			kPFPixelFormatSuite,
			kPFPixelFormatSuiteVersion1,
			out_data);

		PrPixelFormat srcPixelFormat = PrPixelFormat_BGRA_4444_8u;
		PrPixelFormat dstPixelFormat = PrPixelFormat_BGRA_4444_8u;

		pixelFormatSuite->GetPixelFormat(input, &srcPixelFormat);
		pixelFormatSuite->GetPixelFormat(output, &dstPixelFormat);

		return ActuallyRender(
			in_data,
			input,
			out_data,
			output,
			ntscrs_premiere_pixel_format_to_pixel_format(srcPixelFormat),
			ntscrs_premiere_pixel_format_to_pixel_format(dstPixelFormat));
	} else {
		return PF_Err_INVALID_CALLBACK; // we don't support non-SmartFX unless it's Premiere
	}
}

static PF_Err
SmartRender(
	PF_InData				*in_data,
	PF_OutData				*out_data,
	PF_SmartRenderExtra		*extra)

{

	PF_Err			err		= PF_Err_NONE,
					err2 	= PF_Err_NONE;

	PF_EffectWorld	*input_worldP	= NULL,
					*output_worldP  = NULL;

	// checkout input & output buffers.
	ERR((extra->cb->checkout_layer_pixels(in_data->effect_ref, LAYER_INPUT, &input_worldP)));

	ERR(extra->cb->checkout_output(in_data->effect_ref, &output_worldP));

	// TODO: Not sure why this is needed, but sometimes the layer checkouts don't work?
	if (!input_worldP || !output_worldP) {
		return err;
	}

	AEFX_SuiteScoper<PF_WorldSuite2> worldSuite = AEFX_SuiteScoper<PF_WorldSuite2>(in_data,
		kPFWorldSuite,
		kPFWorldSuiteVersion2,
		out_data);

	PF_PixelFormat srcPixelFormat = PrPixelFormat_BGRA_4444_8u;
	PF_PixelFormat dstPixelFormat = PrPixelFormat_BGRA_4444_8u;

	worldSuite->PF_GetPixelFormat(input_worldP, &srcPixelFormat);
	worldSuite->PF_GetPixelFormat(output_worldP, &dstPixelFormat);

	ERR(ActuallyRender(	in_data,
						input_worldP,
						out_data,
						output_worldP,
						ntscrs_world_pixel_format_to_pixel_format(srcPixelFormat),
						ntscrs_world_pixel_format_to_pixel_format(dstPixelFormat)));

	return err;

}

// Map the params that ntsc-rs gives us to AE params.
static PF_Err NtscSetupParams (
	PF_InData* in_data,
	ntscrs_SettingsList settings,
	// Number of params that are actual settings for ntsc-rs (namely, *not* PF_Param_GROUP_START and PF_PARAM_GROUP_END)
	A_long* num_checkoutable_params,
	// All params in the output list. After Effects is apparently incapable of counting the PF_ADD_PARAM calls we make
	// during this callback, and needs us to tell it. It'll helpfully give us an error dialog if we miscount, though!
	// Thanks, Adobe.
	A_long* num_aefx_params,
	// A list of every param that we should check out when we go to render.
	NtscAE_CheckoutableParam* checkoutable_params
) {
	PF_Err err = PF_Err_NONE;
	for (A_long i = 0; i < settings.len; i++) {
		PF_ParamDef def;
		AEFX_CLR_STRUCT(def);

		def.flags = def.ui_flags = def.ui_width = def.ui_height = 0;
		char* tmp_buf = NULL;

		const ntscrs_SettingDescriptor* descriptor = settings.descriptors + i;
		switch (descriptor->kind.tag) {
			case ntscrs_SettingKind_Enumeration: {
				def.param_type = PF_Param_POPUP;
				PF_PopupDef* popup_def = &def.u.pd;

				A_short default_idx = 0;
				// Length of the "|"-separated string that AE wants, including the null byte at the end
				A_long items_len = 0;
				for (A_long i = 0; i < descriptor->kind.enumeration.len; i++) {
					ntscrs_MenuItem option = descriptor->kind.enumeration.options[i];
					// Length of the label itself, plus 1 byte for either the "|" or null terminator
					items_len += option.label_len + 1;

					if (option.index == descriptor->kind.enumeration.default_value) {
						default_idx = i;
					}
				}
				if (items_len == 0) {
					items_len = 1;
				}

				// Generate the menu string by separating all the menu item strings with "|"
				// TODO: sanitize the strings just in case?
				tmp_buf = (char*)malloc(items_len);
				char* cur = tmp_buf;
				for (A_long i = 0; i < descriptor->kind.enumeration.len; i++) {
					ntscrs_MenuItem option = descriptor->kind.enumeration.options[i];
					memcpy(cur, option.label, option.label_len);
					cur += option.label_len;
					if (i == descriptor->kind.enumeration.len - 1) {
						*cur = '\0';
					} else {
						*cur = '|';
						cur++;
					}
				}

				popup_def->u.namesptr = tmp_buf;
				// Menu option indices are offset by 1(???)
				popup_def->dephault = popup_def->value = default_idx + 1;
				popup_def->num_choices = descriptor->kind.enumeration.len;

				break;
			}
			case ntscrs_SettingKind_Percentage: {
				def.param_type = PF_Param_FLOAT_SLIDER;
				PF_FloatSliderDef* slider_def = &def.u.fs_d;

				slider_def->valid_min = slider_def->slider_min = 0.0;
				slider_def->valid_max = slider_def->slider_max = 100.0;

				float default_value = descriptor->kind.percentage.default_value;
				if (descriptor->kind.percentage.logarithmic) {
					default_value = map_logarithmic_inverse(default_value, 0.0, 1.0, LOG_SLIDER_BASE);
				}

				slider_def->value = slider_def->dephault = default_value * 100.0;
				slider_def->precision = 1;
				slider_def->display_flags = PF_ValueDisplayFlag_PERCENT;

				break;
			}
			case ntscrs_SettingKind_IntRange: {
				def.param_type = PF_Param_FLOAT_SLIDER;
				PF_FloatSliderDef* slider_def = &def.u.fs_d;

				slider_def->valid_min = slider_def->slider_min = (float)(descriptor->kind.int_range.min);
				slider_def->valid_max = slider_def->slider_max = (float)(descriptor->kind.int_range.max);
				slider_def->value = slider_def->dephault = (float)(descriptor->kind.int_range.default_value);
				slider_def->precision = 0;

				break;
			}
			case ntscrs_SettingKind_FloatRange: {
				def.param_type = PF_Param_FLOAT_SLIDER;
				PF_FloatSliderDef* slider_def = &def.u.fs_d;

				ntscrs_SettingKind_ntscrs_FloatRange_Body float_range = descriptor->kind.float_range;

				float default_value = float_range.default_value;
				if (float_range.logarithmic) {
					default_value = map_logarithmic_inverse(default_value, float_range.min, float_range.max, LOG_SLIDER_BASE);
				}


				slider_def->valid_min = slider_def->slider_min = descriptor->kind.float_range.min;
				slider_def->valid_max = slider_def->slider_max = descriptor->kind.float_range.max;
				slider_def->value = slider_def->dephault = descriptor->kind.float_range.default_value;
				slider_def->precision = 2;

				break;
			}
			case ntscrs_SettingKind_Boolean: {
				def.param_type = PF_Param_CHECKBOX;
				PF_CheckBoxDef* checkbox_def = &def.u.bd;

				checkbox_def->value = checkbox_def->dephault = descriptor->kind.boolean.default_value;
				checkbox_def->u.nameptr = descriptor->label;
				break;
			}
			case ntscrs_SettingKind_Group: {
				def.param_type = PF_Param_GROUP_START;
				strncpy(def.name, (const char*)(descriptor->label), 32);
				// Give the group start a really high ID. Not sure if group params need IDs, but I learned that if we
				// mess up parameter disk ID assignment (I accidentally assigned the param group ID to the GROUP_START
				// and not the checkbox), AE will *refuse to load any comp that had a previous version of this effect
				// applied on any layer*, and tell the user that the data is corrupt and missing. High stakes!
				def.uu.id = (descriptor->id + 1) | 1024;
				ERR(PF_ADD_PARAM(in_data, -1, &def));
				(*num_aefx_params)++;

				AEFX_CLR_STRUCT(def);
				def.param_type = PF_Param_CHECKBOX;
				PF_CheckBoxDef* checkbox_def = &def.u.bd;
				checkbox_def->value = checkbox_def->dephault = true;
				checkbox_def->u.nameptr = StrID_Param_Group_Enabled;
				def.flags |= PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_SUPERVISE;
				def.uu.id = descriptor->id + 1;
				ERR(PF_ADD_PARAM(in_data, -1, &def));
				checkoutable_params[*num_checkoutable_params] = {descriptor->id, *num_aefx_params};
				(*num_aefx_params)++;
				// We need to increment this *before* descending into the child params
				(*num_checkoutable_params)++;

				NtscSetupParams(in_data, {descriptor->kind.group.children, descriptor->kind.group.len}, num_checkoutable_params, num_aefx_params, checkoutable_params);

				AEFX_CLR_STRUCT(def);
				def.param_type = PF_Param_GROUP_END;
				(*num_aefx_params)++;
				break;
			}
			default: continue;
		}
		// Without this flag, the param will "start collapsed", but become expanded upon any PF_UpdateParamUI call.
		// Thanks a lot, AE.
		def.flags |= PF_ParamFlag_START_COLLAPSED;
		strncpy(def.name, (const char*)(descriptor->label), 32);
		if (descriptor->kind.tag == ntscrs_SettingKind_Group) {
			def.uu.id = (descriptor->id + 1) | 2048;
		} else {
			checkoutable_params[*num_checkoutable_params] = {descriptor->id, *num_aefx_params};
			def.uu.id = descriptor->id + 1;
			(*num_checkoutable_params)++;
			(*num_aefx_params)++;
		}
		ERR(PF_ADD_PARAM(in_data, -1, &def));

		// Now that we've copied the effect data via PF_ADD_PARAM, we can free any temporary data used to construct it
		// (e.g. the string buffer used to construct menu listings).
		if (tmp_buf) {
			free(tmp_buf);
		}
	}

	return err;
}

static PF_Err
ParamsSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_Err			err = PF_Err_NONE;

	AEGP_SuiteHandler suites(in_data->pica_basicP);
	NtscAE_GlobalData* global_data = reinterpret_cast<NtscAE_GlobalData*>(suites.HandleSuite1()->host_lock_handle(out_data->global_data));
	if (!global_data) return PF_Err_BAD_CALLBACK_PARAM;
	global_data->settings = ntscrs_settingslist_create();

	// Map parameter indices to parameter IDs
	NtscAE_CheckoutableParam* checkoutable_params = (NtscAE_CheckoutableParam*)malloc(global_data->settings.total_num_settings * sizeof(NtscAE_CheckoutableParam));

	A_long num_checkoutable_params = 0;
	// We start with 1 param: the input layer
	A_long num_aefx_params = 1;
	NtscSetupParams(in_data, global_data->settings, &num_checkoutable_params, &num_aefx_params, checkoutable_params);
	global_data->num_checkoutable_params = num_checkoutable_params;
	global_data->num_aefx_params = num_aefx_params;
	global_data->checkoutable_params = checkoutable_params;

	out_data->num_params = num_aefx_params;

	suites.HandleSuite1()->host_unlock_handle(out_data->global_data);

	return err;
}

static PF_Err UpdateParameterUI (
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[]
) {
	PF_Err err = PF_Err_NONE;
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	NtscAE_GlobalData* global_data = reinterpret_cast<NtscAE_GlobalData*>(suites.HandleSuite1()->host_lock_handle(out_data->global_data));

	// Update group controls' enabled/disabled state based on whether the group checkbox is checked
	// We need to do this in an UPDATE_PARAMS_UI callback rather than just supervising the checkboxes so that the
	// visibility is properly set when the effect is first loaded (e.g. when loading a comp with it applied to a layer
	// already). Otherwise, controls for a disabled param group wouldn't show as disabled until the user toggled the
	// group's checkbox after loading.
	PF_ParamDef param_to_update;
	A_Boolean last_param_was_group = false;

	// We track the disabled-ness of nested groups using an int as a stack. The lowest bit is 1 if the top-level group
	// is disabled, the next-lowest bit is 1 if the next group down is disabled, and so on. This lets us check if any
	// group in the stack is disabled by checking if disabled_stack != 0. This is why 1 means disabled--we want to
	// disable a param if *any* parent groups are disabled.
	A_long disabled_stack = 0;
	A_long group_depth = 0;

	for (int i = 0; i < global_data->num_aefx_params; i++) {
		param_to_update = *params[i];

		// Group started--next iteration we will check the checkbox state
		if (param_to_update.param_type == PF_Param_GROUP_START) {
			last_param_was_group = true;
			group_depth++;
			continue;
		}

		if (param_to_update.param_type == PF_Param_GROUP_END) {
			disabled_stack &= ~(1 << (group_depth - 1));
			group_depth--;
			continue;
		}

		// We're inside a group--update the UI element from the checkbox value
		if (group_depth != 0) {
			A_Boolean was_enabled = (param_to_update.ui_flags & PF_PUI_DISABLED) == 0;
			A_Boolean enabled = disabled_stack == 0;

			// Skip UI refresh if nothing actually changed
			if (enabled != was_enabled) {
				if (enabled) {
					param_to_update.ui_flags &= ~PF_PUI_DISABLED;
				} else {
					param_to_update.ui_flags |= PF_PUI_DISABLED;
				}

				suites.ParamUtilsSuite3()->PF_UpdateParamUI(in_data->effect_ref, i, &param_to_update);
			}
		}

		if (last_param_was_group) {
			last_param_was_group = false;
			// Sanity check--make sure the group's first item is actually a checkbox
			if (param_to_update.param_type == PF_Param_CHECKBOX) {
				A_Boolean disabled = param_to_update.u.bd.value == 0;
				disabled_stack |= disabled << (group_depth - 1);
			}
		}
	}

	suites.HandleSuite1()->host_unlock_handle(out_data->global_data);

	return err;
}

extern "C" DllExport
PF_Err PluginDataEntryFunction(
	PF_PluginDataPtr inPtr,
	PF_PluginDataCB inPluginDataCallBackPtr,
	SPBasicSuite* inSPBasicSuitePtr,
	const char* inHostName,
	const char* inHostVersion)
{
	PF_Err result = PF_Err_INVALID_CALLBACK;

	result = PF_REGISTER_EFFECT(
		inPtr,
		inPluginDataCallBackPtr,
		StrID_Name, // Name
		StrID_MatchName, // Match Name
		StrID_PluginCategory, // Category
		AE_RESERVED_INFO); // Reserved Info

	return result;
}

PF_Err
EffectMain(
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra)
{
	PF_Err		err = PF_Err_NONE;

	try	{
		switch (cmd) {
			case PF_Cmd_ABOUT:
				err = About(in_data,out_data,params,output);
				break;
			case PF_Cmd_GLOBAL_SETUP:
				err = GlobalSetup(in_data,out_data,params,output);
				break;
			case PF_Cmd_PARAMS_SETUP:
				err = ParamsSetup(in_data,out_data,params,output);
				break;
			case PF_Cmd_RENDER:
				err = Render(in_data, out_data, params, output);
				break;
			case PF_Cmd_SMART_PRE_RENDER:
				err = PreRender(in_data, out_data, (PF_PreRenderExtra*)extra);
				break;
			case PF_Cmd_SMART_RENDER:
				err = SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra);
				break;
			case PF_Cmd_UPDATE_PARAMS_UI:
				err = UpdateParameterUI(in_data, out_data, params);
				break;
		}
	} catch(PF_Err &thrown_err) {
		err = thrown_err;
	}
	return err;
}
