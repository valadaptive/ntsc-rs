#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>


typedef struct ntscrs_Configurator ntscrs_Configurator;

typedef struct ntscrs_DescriptorsById ntscrs_DescriptorsById;

typedef struct ntscrs_MenuItem {
  char *label;
  size_t label_len;
  char *description;
  size_t description_len;
  uint32_t index;
} ntscrs_MenuItem;

typedef enum ntscrs_SettingKind_Tag {
  ntscrs_SettingKind_Enumeration,
  ntscrs_SettingKind_Percentage,
  ntscrs_SettingKind_IntRange,
  ntscrs_SettingKind_FloatRange,
  ntscrs_SettingKind_Boolean,
  ntscrs_SettingKind_Group,
} ntscrs_SettingKind_Tag;

typedef struct ntscrs_SettingKind_ntscrs_Enumeration_Body {
  struct ntscrs_MenuItem *options;
  size_t len;
  uint32_t default_value;
} ntscrs_SettingKind_ntscrs_Enumeration_Body;

typedef struct ntscrs_SettingKind_ntscrs_Percentage_Body {
  bool logarithmic;
  float default_value;
} ntscrs_SettingKind_ntscrs_Percentage_Body;

typedef struct ntscrs_SettingKind_ntscrs_IntRange_Body {
  int32_t min;
  int32_t max;
  int32_t default_value;
} ntscrs_SettingKind_ntscrs_IntRange_Body;

typedef struct ntscrs_SettingKind_ntscrs_FloatRange_Body {
  float min;
  float max;
  bool logarithmic;
  float default_value;
} ntscrs_SettingKind_ntscrs_FloatRange_Body;

typedef struct ntscrs_SettingKind_ntscrs_Boolean_Body {
  bool default_value;
} ntscrs_SettingKind_ntscrs_Boolean_Body;

typedef struct ntscrs_SettingKind_ntscrs_Group_Body {
  struct ntscrs_SettingDescriptor *children;
  size_t len;
  bool default_value;
} ntscrs_SettingKind_ntscrs_Group_Body;

typedef struct ntscrs_SettingKind {
  ntscrs_SettingKind_Tag tag;
  union {
    ntscrs_SettingKind_ntscrs_Enumeration_Body enumeration;
    ntscrs_SettingKind_ntscrs_Percentage_Body percentage;
    ntscrs_SettingKind_ntscrs_IntRange_Body int_range;
    ntscrs_SettingKind_ntscrs_FloatRange_Body float_range;
    ntscrs_SettingKind_ntscrs_Boolean_Body boolean;
    ntscrs_SettingKind_ntscrs_Group_Body group;
  };
} ntscrs_SettingKind;

typedef struct ntscrs_SettingDescriptor {
  char *label;
  size_t label_len;
  char *description;
  size_t description_len;
  struct ntscrs_SettingKind kind;
  uint32_t id;
} ntscrs_SettingDescriptor;

typedef struct ntscrs_SettingsList {
  const struct ntscrs_SettingDescriptor *descriptors;
  size_t len;
  size_t total_num_settings;
  const struct ntscrs_DescriptorsById *by_id;
} ntscrs_SettingsList;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct ntscrs_Configurator *ntscrs_configurator_create(void);

void ntscrs_configurator_free(struct ntscrs_Configurator *configurator_ptr);

void ntscrs_process_yiq(float *y,
                        float *i,
                        float *q,
                        size_t width,
                        size_t height,
                        const struct ntscrs_Configurator *settings,
                        size_t frame_num,
                        uint64_t seed);

bool ntscrs_settings_get_field_bool(struct ntscrs_Configurator *self, uint32_t id);

float ntscrs_settings_get_field_float(struct ntscrs_Configurator *self, uint32_t id);

int32_t ntscrs_settings_get_field_int(struct ntscrs_Configurator *self, uint32_t id);

void ntscrs_settings_set_field_bool(struct ntscrs_Configurator *self, uint32_t id, bool value);

void ntscrs_settings_set_field_float(struct ntscrs_Configurator *self, uint32_t id, float value);

void ntscrs_settings_set_field_int(struct ntscrs_Configurator *self, uint32_t id, int32_t value);

struct ntscrs_SettingsList ntscrs_settingslist_create(void);

void ntscrs_settingslist_free(struct ntscrs_SettingsList self);

const struct ntscrs_SettingDescriptor *ntscrs_settingslist_get_descriptor_by_id(const struct ntscrs_SettingsList *self,
                                                                                uint32_t id);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
