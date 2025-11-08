//! This is used to dynamically inform API consumers of the settings that can be passed to ntsc-rs. This lets various
//! UIs and effect plugins to query this set of settings and display them in their preferred format without having to
//! duplicate a bunch of code.
// TODO: replace with a bunch of metaprogramming macro magic?

use std::{collections::HashMap, error::Error, fmt::Display, ops::RangeInclusive};

use num_enum::TryFromPrimitive;
pub use sval;
pub use sval_json;
use sval_json::{stream_to_fmt_write, stream_to_io_write};
use tinyjson::{InnerAsRef, JsonParseError, JsonValue};

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

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnumValue(pub u32);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnySetting {
    Enum(EnumValue),
    Int(i32),
    Float(f32),
    Bool(bool),
}

impl AnySetting {
    pub fn type_name(&self) -> &'static str {
        match self {
            AnySetting::Enum(_) => "enum",
            AnySetting::Int(_) => "i32 or u32",
            AnySetting::Float(_) => "f32",
            AnySetting::Bool(_) => "bool",
        }
    }
}

pub trait Downcast: Sized {
    fn downcast(value: &AnySetting) -> Option<Self>;
}

impl<U: TryFrom<u32>, T: SettingsEnum + TryFromPrimitive<Primitive = U>> Downcast for T {
    fn downcast(value: &AnySetting) -> Option<Self> {
        match value {
            AnySetting::Enum(e) => T::try_from_primitive(e.0.try_into().ok()?).ok(),
            _ => None,
        }
    }
}

impl Downcast for i32 {
    fn downcast(value: &AnySetting) -> Option<Self> {
        match value {
            AnySetting::Int(i) => Some(*i),
            _ => None,
        }
    }
}

impl Downcast for u32 {
    fn downcast(value: &AnySetting) -> Option<Self> {
        match value {
            AnySetting::Int(i) => Some(*i as u32),
            _ => None,
        }
    }
}

impl Downcast for EnumValue {
    fn downcast(value: &AnySetting) -> Option<Self> {
        match value {
            AnySetting::Enum(e) => Some(*e),
            _ => None,
        }
    }
}

impl Downcast for f32 {
    fn downcast(value: &AnySetting) -> Option<Self> {
        match value {
            AnySetting::Float(f) => Some(*f),
            _ => None,
        }
    }
}

impl Downcast for bool {
    fn downcast(value: &AnySetting) -> Option<Self> {
        match value {
            AnySetting::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

impl<U: Into<u32>, T: SettingsEnum + TryFromPrimitive<Primitive = U> + Into<U>> From<T>
    for AnySetting
{
    fn from(value: T) -> Self {
        Self::Enum(EnumValue(<T as Into<U>>::into(value).into()))
    }
}

impl From<i32> for AnySetting {
    fn from(value: i32) -> Self {
        Self::Int(value)
    }
}

impl From<u32> for AnySetting {
    fn from(value: u32) -> Self {
        Self::Int(value as i32)
    }
}

impl From<EnumValue> for AnySetting {
    fn from(value: EnumValue) -> Self {
        Self::Enum(value)
    }
}

impl From<f32> for AnySetting {
    fn from(value: f32) -> Self {
        Self::Float(value)
    }
}

impl From<bool> for AnySetting {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

pub trait SettingsEnum {}

/// A fixed identifier that points to a given setting. The id and name cannot be changed or reused once created.
#[derive(Debug, Clone)]
pub struct SettingID<T: Settings> {
    pub id: u32,
    pub name: &'static str,
    pub get: fn(settings: &T) -> AnySetting,
    pub set: fn(settings: &mut T, value: AnySetting) -> Result<(), GetSetFieldError>,
}

// We can't use derive here because of the type parameter:
// https://github.com/rust-lang/rust/issues/26925
impl<T: Settings> std::hash::Hash for SettingID<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.name.hash(state);
        self.get.hash(state);
        self.set.hash(state);
    }
}

impl<T: Settings> PartialEq for SettingID<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.name == other.name
    }
}

impl<T: Settings> Eq for SettingID<T> {}

impl<T: Settings> SettingID<T> {
    pub const fn new(
        id: u32,
        name: &'static str,
        get: fn(settings: &T) -> AnySetting,
        set: fn(settings: &mut T, value: AnySetting) -> Result<(), GetSetFieldError>,
    ) -> Self {
        Self { id, name, get, set }
    }
}

#[macro_export]
macro_rules! setting_id {
    ($id:expr, $name:expr, $($field_path:ident).+) => {
        $crate::settings::SettingID::new(
            $id,
            $name,
            |settings| settings.$($field_path).+.into(),
            |settings, value| {
                settings.$($field_path).+ = $crate::settings::Downcast::downcast(&value).ok_or_else(|| $crate::settings::GetSetFieldError::TypeMismatch {
                    actual_type: value.type_name(),
                    requested_type: std::any::type_name_of_val(&settings.$($field_path).+)
                })?;
                Ok(())
            }
        )
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
    Enumeration { options: Vec<MenuItem> },
    /// Range from 0% to 100%.
    Percentage { logarithmic: bool },
    /// Inclusive discrete (integer) range.
    IntRange { range: RangeInclusive<i32> },
    /// Inclusive continuous range.
    FloatRange {
        range: RangeInclusive<f32>,
        logarithmic: bool,
    },
    /// Boolean/checkbox.
    Boolean,
    /// Group of settings, which contains an "enable/disable" checkbox and child settings.
    Group { children: Vec<SettingDescriptor<T>> },
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
            GetSetFieldError::TypeMismatch {
                actual_type,
                requested_type,
            } => write!(
                f,
                "Tried to get or set field with type {requested_type}, but actual type is \
                 {actual_type}"
            ),
            GetSetFieldError::NoSuchID(id) => write!(f, "No such field with ID {id}"),
        }
    }
}

pub trait Settings: Default {
    fn get_field<T: 'static + Downcast>(
        &self,
        id: &SettingID<Self>,
    ) -> Result<T, GetSetFieldError> {
        let value = (id.get)(self);
        Downcast::downcast(&value).ok_or_else(|| GetSetFieldError::TypeMismatch {
            actual_type: value.type_name(),
            requested_type: std::any::type_name::<T>(),
        })
    }

    fn set_field<T: 'static + Into<AnySetting>>(
        &mut self,
        id: &SettingID<Self>,
        value: T,
    ) -> Result<(), GetSetFieldError> {
        (id.set)(self, value.into())
    }

    /// Returns settings which e.g. new presets can be applied on top of without any newly-added settings having an
    /// additional effect on the result. For example, a new setting added to an existing group would probably be set to
    /// 0, whereas an entirely new settings group could have all its settings at nice defaults but simply be disabled.
    /// Settings that have always existed can take on their regular default values, which are subject to change.
    fn legacy_value() -> Self;

    fn setting_descriptors() -> Box<[SettingDescriptor<Self>]>;
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
pub(super) trait GetAndExpect {
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
    pub setting_descriptors: Box<[SettingDescriptor<T>]>,
}

struct SettingsAndList<'a, 'b, T: Settings> {
    settings: &'a T,
    list: &'b SettingsList<T>,
}

impl<T: Settings> sval::Value for SettingsAndList<'_, '_, T> {
    fn stream<'sval, S: sval::Stream<'sval> + ?Sized>(&'sval self, stream: &mut S) -> sval::Result {
        stream.map_begin(None)?;

        for descriptor in self.list.all_descriptors() {
            stream.map_key_begin()?;
            stream.text_begin(Some(descriptor.id.name.len()))?;
            stream.text_fragment(descriptor.id.name)?;
            stream.text_end()?;
            stream.map_key_end()?;

            stream.map_value_begin()?;
            match &descriptor.kind {
                SettingKind::Enumeration { .. } => {
                    stream.u32(
                        self.settings
                            .get_field::<EnumValue>(&descriptor.id)
                            .unwrap()
                            .0,
                    )?;
                }
                SettingKind::Percentage { .. } | SettingKind::FloatRange { .. } => {
                    stream.f32(self.settings.get_field::<f32>(&descriptor.id).unwrap())?;
                }
                SettingKind::IntRange { .. } => {
                    stream.i32(self.settings.get_field::<i32>(&descriptor.id).unwrap())?;
                }
                SettingKind::Boolean | SettingKind::Group { .. } => {
                    stream.bool(self.settings.get_field::<bool>(&descriptor.id).unwrap())?;
                }
            }
            stream.map_value_end()?;
        }

        stream.map_key_begin()?;
        stream.text_begin(Some("version".len()))?;
        stream.text_fragment("version")?;
        stream.text_end()?;
        stream.map_key_end()?;
        stream.map_value_begin()?;
        stream.u32(1)?;
        stream.map_value_end()?;

        stream.map_end()?;

        Ok(())
    }
}

impl<T: Settings> SettingsList<T> {
    /// Construct a list of all the effect settings. This isn't meant to be mutated--you should just create one instance
    /// of this to use for your entire application/plugin.
    pub fn new() -> Self {
        Self {
            setting_descriptors: T::setting_descriptors(),
        }
    }

    /// Convert the settings in the given settings struct to JSON.
    pub fn to_json<'a>(&'a self, settings: &'a T) -> impl sval::Value + 'a {
        SettingsAndList {
            settings,
            list: self,
        }
    }

    pub fn write_json_to_fmt(
        &self,
        settings: &T,
        dest: impl std::fmt::Write,
    ) -> Result<(), sval_json::Error> {
        let json = self.to_json(settings);
        stream_to_fmt_write(dest, json)
    }

    pub fn write_json_to_io(
        &self,
        settings: &T,
        dest: impl std::io::Write,
    ) -> Result<(), sval_json::Error> {
        let json = self.to_json(settings);
        stream_to_io_write(dest, json)
    }

    pub fn to_json_string(&self, settings: &T) -> Result<String, sval_json::Error> {
        let mut s = String::new();
        self.write_json_to_fmt(settings, &mut s)?;
        Ok(s)
    }

    /// Recursive method for reading the settings within a given list of descriptors (either top-level or within a
    /// group) from a given JSON map and using them to update the given settings struct.
    pub(super) fn settings_from_json(
        json: &HashMap<String, JsonValue>,
        descriptors: &[SettingDescriptor<T>],
        settings: &mut T,
    ) -> Result<(), ParseSettingsError> {
        for descriptor in descriptors {
            let key = descriptor.id.name;
            match &descriptor.kind {
                SettingKind::Enumeration { .. } => {
                    json.get_and_expect::<f64>(key)?
                        .map(|n| {
                            settings.set_field::<EnumValue>(&descriptor.id, EnumValue(n as u32))
                        })
                        .transpose()?;
                }
                SettingKind::FloatRange { range, .. } => {
                    json.get_and_expect::<f64>(key)?.map(|n| {
                        settings.set_field::<f32>(
                            &descriptor.id,
                            (n as f32).clamp(*range.start(), *range.end()),
                        )
                    });
                }
                SettingKind::Percentage { .. } => {
                    json.get_and_expect::<f64>(key)?.map(|n| {
                        settings.set_field::<f32>(&descriptor.id, (n as f32).clamp(0.0, 1.0))
                    });
                }
                SettingKind::IntRange { range, .. } => {
                    json.get_and_expect::<f64>(key)?.map(|n| {
                        settings.set_field::<i32>(
                            &descriptor.id,
                            (n as i32).clamp(*range.start(), *range.end()),
                        )
                    });
                }
                SettingKind::Boolean { .. } => {
                    json.get_and_expect::<bool>(key)?
                        .map(|b| settings.set_field::<bool>(&descriptor.id, b));
                }
                SettingKind::Group { children, .. } => {
                    json.get_and_expect::<bool>(key)?
                        .map(|b| settings.set_field::<bool>(&descriptor.id, b));
                    Self::settings_from_json(json, children, settings)?;
                }
            }
        }

        Ok(())
    }

    /// Parse settings from a given string of JSON and return a new settings struct.
    pub fn from_json_generic(&self, json: &str) -> Result<T, ParseSettingsError> {
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

        let mut dst_settings = T::legacy_value();
        Self::settings_from_json(parsed_map, &self.setting_descriptors, &mut dst_settings)?;

        Ok(dst_settings)
    }

    pub fn all_descriptors(&self) -> SettingDescriptors<'_, T> {
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
            path: vec![(&settings_list.setting_descriptors, 0)],
        }
    }
}
