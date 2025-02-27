use std::{borrow::Cow, collections::HashSet};

use eframe::{
    egui::FontData,
    epaint::text::{FontInsert, InsertFontFamily},
};
use log::{debug, warn};

pub fn system_fallback_fonts() -> impl Iterator<Item = FontInsert> {
    let mut collection = fontique::Collection::new(fontique::CollectionOptions {
        shared: false,
        system_fonts: true,
    });

    // Get fallback fonts for CJK.
    // We may also want to try loading Arabic and Devanagari fonts, but egui can't shape them properly right now
    let mut locale_families = Vec::<fontique::FamilyId>::new();
    for script in [b"Hira", b"Kana", b"Hang", b"Hani"] {
        locale_families
            .extend(collection.fallback_families(fontique::FallbackKey::new(script, None)));
    }

    let mut seen_sources = HashSet::new();
    let mut fonts = Vec::new();
    for family_id in locale_families {
        let Some(family) = collection.family(family_id) else {
            continue;
        };
        let Some(font) = family.match_font(
            fontique::Stretch::NORMAL,
            fontique::Style::Normal,
            fontique::Weight::NORMAL,
            false,
        ) else {
            continue;
        };
        let fontique::SourceKind::Path(path) = font.source().kind() else {
            continue;
        };

        // We may get the same font file for multiple scripts
        if !seen_sources.insert(font.source().id()) {
            continue;
        }

        debug!("Loading font {} from {:?}", family.name(), path);
        let font_data = match std::fs::read(path) {
            Ok(font_data) => font_data,
            Err(e) => {
                warn!("{:?}", e);
                continue;
            }
        };

        fonts.push(FontInsert {
            name: family.name().to_owned(),
            data: FontData {
                font: Cow::Owned(font_data),
                index: font.index(),
                tweak: Default::default(),
            },
            families: vec![InsertFontFamily {
                family: eframe::egui::FontFamily::Proportional,
                priority: eframe::epaint::text::FontPriority::Lowest,
            }],
        });
    }

    fonts.into_iter()
}
