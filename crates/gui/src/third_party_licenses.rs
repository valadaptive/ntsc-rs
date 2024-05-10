use std::{collections::HashMap, sync::OnceLock};

use tinyjson::JsonValue;

static THIRD_PARTY_LICENSES_JSON: &'static str = include_str!("../about.json");
pub struct ThirdPartyCrate {
    pub name: String,
    pub version: String,
    pub url: String,
}
pub struct ThirdPartyLicense {
    pub name: String,
    pub text: String,
    pub used_by: Vec<ThirdPartyCrate>,
}
static THIRD_PARTY_LICENSES: OnceLock<Vec<ThirdPartyLicense>> = OnceLock::new();

pub fn get_third_party_licenses() -> &'static [ThirdPartyLicense] {
    THIRD_PARTY_LICENSES.get_or_init(|| {
        let mut licenses = Vec::<ThirdPartyLicense>::new();
        let json = THIRD_PARTY_LICENSES_JSON.parse::<JsonValue>().unwrap();
        let json = json.get::<HashMap<_, _>>().unwrap();
        let json_licenses = json.get("licenses").unwrap().get::<Vec<_>>().unwrap();
        for license in json_licenses {
            let license = license.get::<HashMap<_, _>>().unwrap();
            let used_by = license
                .get("used_by")
                .unwrap()
                .get::<Vec<_>>()
                .unwrap()
                .iter()
                .map(|used_by| {
                    let used_by = used_by.get::<HashMap<_, _>>().unwrap();
                    let krate = used_by
                        .get("crate")
                        .unwrap()
                        .get::<HashMap<_, _>>()
                        .unwrap();
                    let name = krate.get("name").unwrap().get::<String>().unwrap();
                    let version = krate
                        .get("version")
                        .unwrap()
                        .get::<String>()
                        .unwrap()
                        .clone();
                    ThirdPartyCrate {
                        name: name.clone(),
                        version,
                        url: krate
                            .get("repository")
                            .and_then(|repo| repo.get::<String>())
                            .cloned()
                            .unwrap_or_else(|| format!("https://crates.io/crates/{name}")),
                    }
                })
                .collect::<Vec<_>>();

            licenses.push(ThirdPartyLicense {
                name: license
                    .get("name")
                    .unwrap()
                    .get::<String>()
                    .unwrap()
                    .clone(),
                text: license
                    .get("text")
                    .unwrap()
                    .get::<String>()
                    .unwrap()
                    .clone(),
                used_by,
            })
        }
        licenses
    })
}
