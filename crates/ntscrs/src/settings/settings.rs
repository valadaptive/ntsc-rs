use std::{
    collections::HashMap, error::Error, fmt::Display, marker::PhantomData, ops::RangeInclusive,
};

use tinyjson::{InnerAsRef, JsonParseError, JsonValue};

/// This is used to dynamically inform API consumers of the settings that can be passed to ntsc-rs. This lets various
/// UIs and effect plugins to query this set of settings and display them in their preferred format without having to
/// duplicate a bunch of code.
// TODO: replace with a bunch of metaprogramming macro magic?

// These are the individual setting definitions. The descriptions of what they do are included below, so I mostly won't
// repeat them here.

/// The "full settings" equivalent of an Option<T> for an optionally-disabled section of the settings.
#[derive(Debug, Clone, PartialEq)]
pub struct SettingsBlock<T> {
    pub enabled: bool,
    pub settings: T,
}

impl<T: Default + Clone> From<&Option<T>> for SettingsBlock<T> {
    fn from(opt: &Option<T>) -> Self {
        Self {
            enabled: opt.is_some(),
            settings: match opt {
                Some(v) => v.clone(),
                None => T::default(),
            },
        }
    }
}

impl<T: Default> From<Option<T>> for SettingsBlock<T> {
    fn from(opt: Option<T>) -> Self {
        Self {
            enabled: opt.is_some(),
            settings: opt.unwrap_or_default(),
        }
    }
}

impl<T> From<SettingsBlock<T>> for Option<T> {
    fn from(value: SettingsBlock<T>) -> Self {
        if value.enabled {
            Some(value.settings)
        } else {
            None
        }
    }
}

impl<T: Clone> From<&SettingsBlock<T>> for Option<T> {
    fn from(value: &SettingsBlock<T>) -> Self {
        if value.enabled {
            Some(value.settings.clone())
        } else {
            None
        }
    }
}

impl<T: Default> Default for SettingsBlock<T> {
    fn default() -> Self {
        Self {
            enabled: true,
            settings: T::default(),
        }
    }
}

/// A fixed identifier that points to a given setting. The id and name cannot be changed or reused once created.
#[derive(Debug, PartialEq, Eq)]
pub struct SettingID<T: Settings> {
    pub id: u32,
    pub name: &'static str,
    settings: PhantomData<fn(&()) -> &T>,
}

// We can't use derive here because of the type parameter:
// https://github.com/rust-lang/rust/issues/26925
impl<T: Settings> std::hash::Hash for SettingID<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.name.hash(state);
    }
}
impl<T: Settings> Clone for SettingID<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: Settings> Copy for SettingID<T> {}

impl<T: Settings> SettingID<T> {
    pub const fn new(id: u32, name: &'static str) -> Self {
        Self {
            id,
            name,
            settings: PhantomData,
        }
    }
}

// These macros are used to implement getting and setting various fields on the settings struct based on `SettingID`s.
// Enums require special handling because ToPrimitive and FromPrimitive are used for conversion there, and those traits
// are not object-safe (or otherwise have various ?Sized issues). For all other setting types, we can just use Any to
// do dynamic typing.

#[macro_export]
macro_rules! get_field_ref_impl {
    ($($field_path:ident).+) => {
        {
            let type_name = std::any::type_name_of_val(&$($field_path).+);
            (&$($field_path).+ as &dyn std::any::Any)
                .downcast_ref()
                .ok_or_else(|| $crate::settings::GetSetFieldError::TypeMismatch {
                    actual_type: type_name,
                    requested_type: std::any::type_name::<T>()
                })
        }
    };

    ($($field_path:ident).+, IS_AN_ENUM) => {
        Err($crate::settings::GetSetFieldError::TypeMismatch {
            actual_type: std::any::type_name_of_val(&$($field_path).+),
            requested_type: std::any::type_name::<T>()
        })
    };
}

#[macro_export]
macro_rules! get_field_mut_impl {
    ($($field_path:ident).+) => {
        {
            let type_name = std::any::type_name_of_val(&$($field_path).+);
            (&mut $($field_path).+ as &mut dyn std::any::Any)
                .downcast_mut()
                .ok_or_else(|| $crate::settings::GetSetFieldError::TypeMismatch {
                    actual_type: type_name,
                    requested_type: std::any::type_name::<T>()
                })
        }
    };

    ($($field_path:ident).+, IS_AN_ENUM) => {
        {
            Err($crate::settings::GetSetFieldError::TypeMismatch {
                actual_type: std::any::type_name_of_val(&$($field_path).+),
                requested_type: std::any::type_name::<T>()
            })
        }
    };
}

#[macro_export]
macro_rules! get_field_enum_impl {
    ($($field_path:ident).+) => {
        {
            let type_name = std::any::type_name_of_val(&$($field_path).+);
            Err($crate::settings::GetSetFieldError::TypeMismatch { actual_type: type_name, requested_type: "enum" })
        }
    };

    ($($field_path:ident).+, IS_AN_ENUM) => {
        {
            use num_traits::ToPrimitive;
            Ok($($field_path).+.to_u32().expect("enum fields should be representable as u32"))
        }
    };
}

#[macro_export]
macro_rules! set_field_enum_impl {
    ($value:ident, $($field_path:ident).+) => {
        {
            let type_name = std::any::type_name_of_val(&$($field_path).+);
            Err($crate::settings::GetSetFieldError::TypeMismatch { actual_type: type_name, requested_type: "enum" })
        }
    };

    ($value:ident, $($field_path:ident).+, IS_AN_ENUM) => {
        {
            $($field_path).+ = num_traits::FromPrimitive::from_u32($value).expect("enum fields should be representable as u32");
            Ok(())
        }
    };
}

#[macro_export]
macro_rules! impl_settings_for {
    ($item:ty, $(($field_setting_id:path, $($field_path:ident).+$(, $is_enum:tt)?)),+$(,)?) => {
        impl $crate::settings::Settings for $item {
            fn get_field_mut<T: 'static>(&mut self, id: &$crate::settings::SettingID<Self>) -> Result<&mut T, $crate::settings::GetSetFieldError> {
                match id {
                    $(&$field_setting_id => $crate::get_field_mut_impl!(self.$($field_path).+$(, $is_enum)?),)+
                    _ => Err($crate::settings::GetSetFieldError::NoSuchID(id.name))
                }
            }

            fn get_field_ref<T: 'static>(&self, id: &$crate::settings::SettingID<Self>) -> Result<&T, $crate::settings::GetSetFieldError> {
                match id {
                    $(&$field_setting_id => $crate::get_field_ref_impl!(self.$($field_path).+$(, $is_enum)?),)+
                    _ => Err($crate::settings::GetSetFieldError::NoSuchID(id.name))
                }
            }

            fn get_field_enum(&self, id: &$crate::settings::SettingID<Self>) -> Result<u32, $crate::settings::GetSetFieldError> {
                match id {
                    $(&$field_setting_id => $crate::get_field_enum_impl!(self.$($field_path).+$(, $is_enum)?),)+
                    _ => Err($crate::settings::GetSetFieldError::NoSuchID(id.name))
                }
            }

            fn set_field_enum(&mut self, id: &$crate::settings::SettingID<Self>, value: u32) -> Result<(), $crate::settings::GetSetFieldError> {
                match id {
                    $(&$field_setting_id => $crate::set_field_enum_impl!(value, self.$($field_path).+$(, $is_enum)?),)+
                    _ => Err($crate::settings::GetSetFieldError::NoSuchID(id.name))
                }
            }
        }
    }
}

/// Menu item for a SettingKind::Enumeration.
#[derive(Debug, Clone)]
pub struct MenuItem {
    pub label: &'static str,
    pub description: Option<&'static str>,
    pub index: u32,
}

/// All of the types a setting can take. API consumers can map this to the UI elements available in whatever they're
/// porting ntsc-rs to.
#[derive(Debug, Clone)]
pub enum SettingKind<T: Settings> {
    /// Selection of specific options, preferably in a specific order.
    Enumeration {
        options: Vec<MenuItem>,
        default_value: u32,
    },
    /// Range from 0% to 100%.
    Percentage {
        logarithmic: bool,
        default_value: f32,
    },
    /// Inclusive discrete (integer) range.
    IntRange {
        range: RangeInclusive<i32>,
        default_value: i32,
    },
    /// Inclusive continuous range.
    FloatRange {
        range: RangeInclusive<f32>,
        logarithmic: bool,
        default_value: f32,
    },
    /// Boolean/checkbox.
    Boolean { default_value: bool },
    /// Group of settings, which contains an "enable/disable" checkbox and child settings.
    Group {
        children: Vec<SettingDescriptor<T>>,
        default_value: bool,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum GetSetFieldError {
    TypeMismatch {
        actual_type: &'static str,
        requested_type: &'static str,
    },
    NoSuchID(&'static str),
}

impl Display for GetSetFieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GetSetFieldError::TypeMismatch { actual_type, requested_type } => write!(f, "Tried to get or set field with type {requested_type}, but actual type is {actual_type}"),
            GetSetFieldError::NoSuchID(id) => write!(f, "No such field with ID {id}"),
        }
    }
}

pub trait Settings: Default {
    fn get_field_bool(&self, id: &SettingID<Self>) -> Result<bool, GetSetFieldError> {
        self.get_field_ref(id).copied()
    }
    fn set_field_bool(
        &mut self,
        id: &SettingID<Self>,
        value: bool,
    ) -> Result<(), GetSetFieldError> {
        *self.get_field_mut(id)? = value;
        Ok(())
    }
    fn get_field_float(&self, id: &SettingID<Self>) -> Result<f32, GetSetFieldError> {
        self.get_field_ref(id).copied()
    }
    fn set_field_float(
        &mut self,
        id: &SettingID<Self>,
        value: f32,
    ) -> Result<(), GetSetFieldError> {
        *self.get_field_mut(id)? = value;
        Ok(())
    }
    fn get_field_int(&self, id: &SettingID<Self>) -> Result<i32, GetSetFieldError> {
        self.get_field_ref::<i32>(id)
            .copied()
            .or_else(|_| self.get_field_ref::<u32>(id).copied().map(|v| v as i32))
    }
    fn set_field_int(&mut self, id: &SettingID<Self>, value: i32) -> Result<(), GetSetFieldError> {
        if let Ok(field) = self.get_field_mut::<i32>(id) {
            *field = value;
            return Ok(());
        }

        match self.get_field_mut::<u32>(id) {
            Ok(field) => {
                *field = value as u32;
                Ok(())
            }

            Err(GetSetFieldError::TypeMismatch { actual_type, .. }) => {
                Err(GetSetFieldError::TypeMismatch {
                    actual_type,
                    requested_type: "i32 or u32",
                })
            }

            Err(e) => Err(e),
        }
    }
    fn get_field_enum(&self, id: &SettingID<Self>) -> Result<u32, GetSetFieldError>;
    fn set_field_enum(&mut self, id: &SettingID<Self>, value: u32) -> Result<(), GetSetFieldError>;

    fn get_field_ref<T: 'static>(&self, id: &SettingID<Self>) -> Result<&T, GetSetFieldError>;
    fn get_field_mut<T: 'static>(
        &mut self,
        id: &SettingID<Self>,
    ) -> Result<&mut T, GetSetFieldError>;
}

/// A single setting, which includes the data common to all settings (its name, optional description/tooltip, and ID)
/// along with a SettingKind which contains data specific to the type of setting.
#[derive(Debug, Clone)]
pub struct SettingDescriptor<T: Settings> {
    pub label: &'static str,
    pub description: Option<&'static str>,
    pub kind: SettingKind<T>,
    pub id: SettingID<T>,
}

#[derive(Debug)]
pub enum ParseSettingsError {
    InvalidJSON(JsonParseError),
    MissingField { field: &'static str },
    WrongApplication,
    UnsupportedVersion { version: f64 },
    InvalidSettingType { key: String, expected: &'static str },
    GetSetField(GetSetFieldError),
}

impl Display for ParseSettingsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseSettingsError::InvalidJSON(e) => e.fmt(f),
            ParseSettingsError::MissingField { field } => {
                write!(f, "Missing field: {}", field)
            }
            ParseSettingsError::WrongApplication => {
                write!(f, "ntscQT presets are not supported")
            }
            ParseSettingsError::UnsupportedVersion { version } => {
                write!(f, "Unsupported version: {}", version)
            }
            ParseSettingsError::InvalidSettingType { key, expected } => {
                write!(f, "Setting {} is not a(n) {}", key, expected)
            }
            ParseSettingsError::GetSetField(e) => {
                write!(f, "Error getting or setting field: {}", e)
            }
        }
    }
}

impl Error for ParseSettingsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl From<JsonParseError> for ParseSettingsError {
    fn from(err: JsonParseError) -> Self {
        Self::InvalidJSON(err)
    }
}

impl From<GetSetFieldError> for ParseSettingsError {
    fn from(err: GetSetFieldError) -> Self {
        Self::GetSetField(err)
    }
}

/// Convenience trait for asserting the "shape" of the JSON we're parsing is what we expect.
trait GetAndExpect {
    fn get_and_expect<T: InnerAsRef + Clone>(
        &self,
        key: &str,
    ) -> Result<Option<T>, ParseSettingsError>;
}

impl GetAndExpect for HashMap<String, JsonValue> {
    fn get_and_expect<T: InnerAsRef + Clone>(
        &self,
        key: &str,
    ) -> Result<Option<T>, ParseSettingsError> {
        self.get(key)
            .map(|v| {
                v.get::<T>()
                    .cloned()
                    .ok_or_else(|| ParseSettingsError::InvalidSettingType {
                        key: key.to_owned(),
                        expected: std::any::type_name::<T>(),
                    })
            })
            .transpose()
    }
}

/// Introspectable list of settings and their types and ranges.
#[derive(Debug, Clone)]
pub struct SettingsList<T: Settings> {
    pub settings: Box<[SettingDescriptor<T>]>,
}

impl<T: Settings> SettingsList<T> {
    /// Recursive method for writing the settings within a given list of descriptors (either top-level or within a
    /// group) to a given JSON map.
    fn settings_to_json(
        dst: &mut HashMap<String, JsonValue>,
        descriptors: &[SettingDescriptor<T>],
        settings: &T,
    ) {
        for descriptor in descriptors {
            let value = match &descriptor.kind {
                SettingKind::Enumeration { .. } => {
                    JsonValue::Number(settings.get_field_enum(&descriptor.id).unwrap() as f64)
                }
                SettingKind::Percentage { .. } | SettingKind::FloatRange { .. } => {
                    JsonValue::Number(settings.get_field_float(&descriptor.id).unwrap() as f64)
                }
                SettingKind::IntRange { .. } => {
                    JsonValue::Number(settings.get_field_int(&descriptor.id).unwrap() as f64)
                }
                SettingKind::Boolean { .. } => {
                    JsonValue::Boolean(settings.get_field_bool(&descriptor.id).unwrap())
                }
                SettingKind::Group { children, .. } => {
                    Self::settings_to_json(dst, children, settings);
                    JsonValue::Boolean(settings.get_field_bool(&descriptor.id).unwrap())
                }
            };

            dst.insert(descriptor.id.name.to_string(), value);
        }
    }

    /// Convert the settings in the given settings struct to JSON.
    pub fn to_json(&self, settings: &T) -> JsonValue {
        let mut dst_map = HashMap::<String, JsonValue>::new();
        Self::settings_to_json(&mut dst_map, &self.settings, settings);

        dst_map.insert("version".to_string(), JsonValue::Number(1.0));

        JsonValue::Object(dst_map)
    }

    /// Recursive method for reading the settings within a given list of descriptors (either top-level or within a
    /// group) from a given JSON map and using them to update the given settings struct.
    fn settings_from_json(
        json: &HashMap<String, JsonValue>,
        descriptors: &[SettingDescriptor<T>],
        settings: &mut T,
    ) -> Result<(), ParseSettingsError> {
        for descriptor in descriptors {
            let key = descriptor.id.name;
            match &descriptor.kind {
                SettingKind::Enumeration { .. } => {
                    json.get_and_expect::<f64>(key)?
                        .map(|n| settings.set_field_enum(&descriptor.id, n as u32))
                        .transpose()?;
                }
                SettingKind::FloatRange { range, .. } => {
                    json.get_and_expect::<f64>(key)?.map(|n| {
                        settings.set_field_float(
                            &descriptor.id,
                            (n as f32).clamp(*range.start(), *range.end()),
                        )
                    });
                }
                SettingKind::Percentage { .. } => {
                    json.get_and_expect::<f64>(key)?.map(|n| {
                        settings.set_field_float(&descriptor.id, (n as f32).clamp(0.0, 1.0))
                    });
                }
                SettingKind::IntRange { range, .. } => {
                    json.get_and_expect::<f64>(key)?.map(|n| {
                        settings.set_field_int(
                            &descriptor.id,
                            (n as i32).clamp(*range.start(), *range.end()),
                        )
                    });
                }
                SettingKind::Boolean { .. } => {
                    json.get_and_expect::<bool>(key)?
                        .map(|b| settings.set_field_bool(&descriptor.id, b));
                }
                SettingKind::Group { children, .. } => {
                    json.get_and_expect::<bool>(key)?
                        .map(|b| settings.set_field_bool(&descriptor.id, b));
                    Self::settings_from_json(json, children, settings)?;
                }
            }
        }

        Ok(())
    }

    /// Parse settings from a given string of JSON and return a new settings struct.
    pub fn from_json(&self, json: &str) -> Result<T, ParseSettingsError> {
        let parsed = json.parse::<JsonValue>()?;

        let parsed_map = parsed.get::<HashMap<_, _>>().ok_or_else(|| {
            ParseSettingsError::InvalidSettingType {
                key: "<root>".to_string(),
                expected: "object",
            }
        })?;

        let version = parsed_map
            .get_and_expect::<f64>("version")?
            .ok_or_else(|| {
                // Detect if the user is trying to import an ntscQT preset, and display a specific error if so
                if parsed_map.contains_key("_composite_preemphasis") {
                    ParseSettingsError::WrongApplication
                } else {
                    ParseSettingsError::MissingField { field: "version" }
                }
            })?;
        if version != 1.0 {
            return Err(ParseSettingsError::UnsupportedVersion { version });
        }

        let mut dst_settings = T::default();
        Self::settings_from_json(parsed_map, &self.settings, &mut dst_settings)?;

        Ok(dst_settings)
    }

    pub fn all_descriptors(&self) -> SettingDescriptors<T> {
        SettingDescriptors::new(self)
    }
}

/// Iterator over all setting descriptors (nested or not) within a given settings list in depth-first order.
pub struct SettingDescriptors<'a, T: Settings> {
    path: Vec<(&'a [SettingDescriptor<T>], usize)>,
}

impl<'a, T: Settings> Iterator for SettingDescriptors<'a, T> {
    type Item = &'a SettingDescriptor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (leaf, index) = self.path.last_mut()?;

            let setting = leaf.get(*index);
            match setting {
                Some(desc) => {
                    *index += 1;
                    // Increment the index of the *current* path node and then recurse into the group. This means that
                    // it'll point to the node after the group once we're finished processing the group.
                    if let SettingKind::Group { children, .. } = &desc.kind {
                        self.path.push((children.as_slice(), 0));
                    }
                    return Some(desc);
                }
                None => {
                    // If the index is pointing one past the end of the list, we traverse upwards (and do so until we
                    // reach the next setting or the end of the top-level list).
                    self.path.pop();
                }
            }
        }
    }
}

impl<'a, T: Settings> SettingDescriptors<'a, T> {
    fn new(settings_list: &'a SettingsList<T>) -> Self {
        Self {
            path: vec![(&settings_list.settings, 0)],
        }
    }
}
