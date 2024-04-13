/*******************************************************************/
/*                                                                 */
/*                      ADOBE CONFIDENTIAL                         */
/*                   _ _ _ _ _ _ _ _ _ _ _ _ _                     */
/*                                                                 */
/* Copyright 2007 Adobe Systems Incorporated                       */
/* All Rights Reserved.                                            */
/*                                                                 */
/* NOTICE:  All information contained herein is, and remains the   */
/* property of Adobe Systems Incorporated and its suppliers, if    */
/* any.  The intellectual and technical concepts contained         */
/* herein are proprietary to Adobe Systems Incorporated and its    */
/* suppliers and may be covered by U.S. and Foreign Patents,       */
/* patents in process, and are protected by trade secret or        */
/* copyright law.  Dissemination of this information or            */
/* reproduction of this material is strictly forbidden unless      */
/* prior written permission is obtained from Adobe Systems         */
/* Incorporated.                                                   */
/*                                                                 */
/*******************************************************************/

#pragma once

#ifndef NtscRS_AE_H
#define NtscRS_AE_H

/*	Ensures AE_Effect.h provides us with 16bpc pixels */

#define PF_DEEP_COLOR_AWARE 1

#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectCBSuites.h"
#include "String_Utils.h"
#include "AE_GeneralPlug.h"
#include "AEGP_SuiteHandler.h"
#include "AEFX_SuiteHelper.h"

#include <ntscrs.h>

#ifdef AE_OS_WIN
	#include <Windows.h>
#endif


// Versioning information

// For future reference:
// https://community.adobe.com/t5/after-effects-discussions/pipl-and-code-version-mismatch-warning/m-p/5531276
// RESOURCE_VERSION =
// (MAJOR_VERSION << 19) +
// (MINOR_VERSION << 15) +
// (BUG_VERSION << 11) +
// (STAGE_VERSION << 9) +
// BUILD_VERSION
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	2
#define	BUG_VERSION		2
#define	STAGE_VERSION	PF_Stage_RELEASE
#define	BUILD_VERSION	1

static const char* StrID_Name = "NTSC-rs",
	*StrID_MatchName = "ntsc-rs",
	*StrID_PluginCategory = "Stylize",
	*StrID_Description = "NTSC and VHS simulation.",
	*StrID_Err_LoadSuite = "Error loading suite.",
	*StrID_Err_FreeSuite = "Error releasing suite.",
	*StrID_Param_Group_Enabled = "Enabled";

// Parameter ID for the layer input. Always 0.
#define LAYER_INPUT 0

extern "C" {
	DllExport
	PF_Err
	EffectMain(
		PF_Cmd			cmd,
		PF_InData		*in_data,
		PF_OutData		*out_data,
		PF_ParamDef		*params[],
		PF_LayerDef		*output,
		void			*extra);
}

// A parameter that we can check out and contributes to the effect settings, e.g. not a GROUP_START or GROUP_END.
typedef struct {
	// The ntscrs ID of this parameter.
	uint32_t id;
	// The index of this parameter in the params list, which includes offsets from GROUP_START and GROUP_END
	A_long index;
} NtscAE_CheckoutableParam;

typedef struct {
	ntscrs_SettingsList settings;
	A_long num_checkoutable_params;
	A_long num_aefx_params; // in_data and out_data both seem to have num_params = 0 in update params callback
	NtscAE_CheckoutableParam* checkoutable_params;
} NtscAE_GlobalData;

// We need to handle the "use field" setting ourselves, so define the setting ID here.
// It's fine to hardcode--if these IDs ever changed, it'd break a bunch of *other* stuff even worse.
#define NTSCRS_USE_FIELD 30

typedef enum {
	NTSCRS_USE_FIELD_ALTERNATING = 0,
	NTSCRS_USE_FIELD_UPPER = 1,
	NTSCRS_USE_FIELD_LOWER = 2,
	NTSCRS_USE_FIELD_BOTH = 3,
	NTSCRS_USE_FIELD_INTERLEAVED_UPPER = 4,
	NTSCRS_USE_FIELD_INTERLEAVED_LOWER = 5
} ntscrs_use_field;

#endif // NtscRS_AE_H
