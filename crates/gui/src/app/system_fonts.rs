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

    let mut seen_sources = HashSet::new();
    let mut fonts = Vec::new();

    let mut load_font = |collection: &mut fontique::Collection,
                         family_id: fontique::FamilyId,
                         script: &[u8; 4]|
     -> bool {
        let Some(family) = collection.family(family_id) else {
            return false;
        };
        let Some(font) = family.match_font(
            fontique::FontWidth::NORMAL,
            fontique::FontStyle::Normal,
            fontique::FontWeight::NORMAL,
            false,
        ) else {
            return false;
        };
        let fontique::SourceKind::Path(path) = font.source().kind() else {
            return false;
        };

        // We may get the same font file for multiple scripts
        if !seen_sources.insert(font.source().id()) {
            debug!(
                "Skipping already-loaded font {} from {:?} for {}",
                family.name(),
                path,
                String::from_utf8_lossy(script)
            );
            return true;
        }

        debug!(
            "Loading font {} from {:?} for {}",
            family.name(),
            path,
            String::from_utf8_lossy(script)
        );
        let font_data = match std::fs::read(path) {
            Ok(font_data) => font_data,
            Err(e) => {
                warn!("{:?}", e);
                return false;
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
        true
    };

    let mut load_fonts = |collection: &mut fontique::Collection, script: &[u8; 4]| -> bool {
        let family_ids = collection
            .fallback_families(fontique::FallbackKey::new(script, None))
            .collect::<Vec<_>>();
        let mut any_loaded = false;
        for family_id in family_ids {
            any_loaded |= load_font(collection, family_id, script);
        }
        any_loaded
    };

    // Get fallback fonts for CJK.
    // We may also want to try loading Arabic and Devanagari fonts, but egui can't shape them properly right now
    for script in [b"Hira", b"Kana", b"Hang", b"Hani", b"Hans", b"Hant"] {
        load_fonts(&mut collection, script);
    }

    fonts.into_iter()
}
