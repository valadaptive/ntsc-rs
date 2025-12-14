extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    AngleBracketedGenericArguments, Attribute, Expr, FieldValue, Fields, GenericArgument, Ident,
    ItemStruct, Member, PathArguments, PathSegment, Token, Type, TypePath, parse_macro_input,
    parse2, spanned::Spanned,
};

fn gen_fields(
    original: &ItemStruct,
) -> (
    Fields,
    Vec<FieldValue>,
    Vec<FieldValue>,
    Vec<FieldValue>,
    Vec<FieldValue>,
) {
    let mut fields = original.fields.clone();
    let mut from_ref_fields = Vec::<FieldValue>::new();
    let mut from_owned_fields = Vec::<FieldValue>::new();
    let mut to_ref_fields = Vec::<FieldValue>::new();
    let mut to_owned_fields = Vec::<FieldValue>::new();

    for field in fields.iter_mut() {
        let name = field.ident.as_ref().unwrap();
        let mut ts_for_ref = quote! {value.#name};
        let mut ts_for_owned = quote! {value.#name};
        let mut ts_to_ref = quote! {value.#name};
        let mut ts_to_owned = quote! {value.#name};
        let mut new_attrs = Vec::<Attribute>::new();
        for attr in &mut field.attrs {
            if attr.path().is_ident("settings_block") {
                let Type::Path(TypePath { path: p, .. }) = &mut field.ty else {
                    continue;
                };

                let Some(segment) = p.segments.first_mut() else {
                    continue;
                };

                let arg_ident: Result<syn::Ident, _> = attr.parse_args();
                let is_nested_fullsettings = match arg_ident {
                    Ok(ident) => ident == "nested",
                    Err(_) => false,
                };

                let (to_fullsettings_ident, from_fullsettings) = if is_nested_fullsettings {
                    let &mut PathSegment {
                        arguments:
                            PathArguments::AngleBracketed(AngleBracketedGenericArguments {
                                args: ref mut arguments,
                                ..
                            }),
                        ..
                    } = segment
                    else {
                        panic!("Expected settings_block to be an Option<...>");
                    };
                    let Some(&mut GenericArgument::Type(Type::Path(TypePath {
                        ref mut path, ..
                    }))) = arguments.first_mut()
                    else {
                        panic!("Expected settings_block to be an Option<...>");
                    };
                    let Some(&mut PathSegment { ref mut ident, .. }) = path.segments.first_mut()
                    else {
                        panic!("Expected settings_block to be an Option<...>");
                    };

                    let fullsettings_ident =
                        Ident::new(&(ident.to_string() + "FullSettings"), ident.span());

                    let old_ident = ident.clone();
                    *ident = fullsettings_ident.clone();

                    (
                        quote! {value.#name.as_ref().map(#fullsettings_ident::from)},
                        quote! {
                            if value.#name.enabled {
                                Some(#old_ident::from(&value.#name.settings))
                            } else {
                                None
                            }
                        },
                    )
                } else {
                    (quote! {value.#name}, quote! {Option::from(&value.#name)})
                };

                segment.ident = Ident::new("SettingsBlock", segment.ident.span());
                ts_for_ref = quote! {SettingsBlock::from(&#to_fullsettings_ident)};
                ts_for_owned = quote! {SettingsBlock::from(#to_fullsettings_ident)};
                ts_to_ref = from_fullsettings.clone();
                ts_to_owned = from_fullsettings.clone();
            } else {
                new_attrs.push(attr.clone());
            }
        }
        field.attrs = new_attrs;

        let expr_for_ref: Expr = parse2(ts_for_ref).unwrap();
        let expr_for_owned: Expr = parse2(ts_for_owned).unwrap();
        let expr_to_ref: Expr = parse2(ts_to_ref).unwrap();
        let expr_to_owned: Expr = parse2(ts_to_owned).unwrap();
        from_ref_fields.push(FieldValue {
            attrs: vec![],
            member: Member::Named(field.ident.as_ref().unwrap().clone()),
            colon_token: Some(Token![:](expr_for_ref.span())),
            expr: expr_for_ref,
        });
        from_owned_fields.push(FieldValue {
            attrs: vec![],
            member: Member::Named(field.ident.as_ref().unwrap().clone()),
            colon_token: Some(Token![:](expr_for_owned.span())),
            expr: expr_for_owned,
        });
        to_ref_fields.push(FieldValue {
            attrs: vec![],
            member: Member::Named(field.ident.as_ref().unwrap().clone()),
            colon_token: Some(Token![:](expr_to_ref.span())),
            expr: expr_to_ref,
        });
        to_owned_fields.push(FieldValue {
            attrs: vec![],
            member: Member::Named(field.ident.as_ref().unwrap().clone()),
            colon_token: Some(Token![:](expr_to_owned.span())),
            expr: expr_to_owned,
        });
    }

    (
        fields,
        from_ref_fields,
        from_owned_fields,
        to_ref_fields,
        to_owned_fields,
    )
}

fn generate(original: &ItemStruct) -> proc_macro2::TokenStream {
    let mut full_struct = original.clone();
    let old_ident = &original.ident;
    let new_ident = &Ident::new(
        &(original.ident.to_string() + "FullSettings"),
        original.ident.span(),
    );
    full_struct.ident = new_ident.clone();

    let (new_fields, from_ref_fields, from_owned_fields, to_ref_fields, to_owned_fields) =
        gen_fields(original);

    full_struct.fields = new_fields;

    quote! {
        #full_struct

        impl From<&#old_ident> for #new_ident {
            fn from(value: &#old_ident) -> Self {
                #new_ident {
                    #(#from_ref_fields),*
                }
            }
        }

        impl From<#old_ident> for #new_ident {
            fn from(value: #old_ident) -> Self {
                #new_ident {
                    #(#from_owned_fields),*
                }
            }
        }

        impl From<&#new_ident> for #old_ident {
            fn from(value: &#new_ident) -> Self {
                #old_ident {
                    #(#to_ref_fields),*
                }
            }
        }

        impl From<#new_ident> for #old_ident {
            fn from(value: #new_ident) -> Self {
                #old_ident {
                    #(#to_owned_fields),*
                }
            }
        }

        impl Default for #new_ident {
            fn default() -> Self {
                Self::from(#old_ident::default())
            }
        }
    }
}

/// Generates a version of this settings block where all Option fields marked as `#[settings_block]` are replaced with
/// versions of those fields that always persist the "Some" values. This is useful for e.g. keeping around the settings'
/// state in UI even if those settings are disabled at the moment.
#[proc_macro_derive(FullSettings, attributes(settings_block))]
pub fn full_settings(item: TokenStream) -> TokenStream {
    let item: ItemStruct = parse_macro_input!(item);

    let full_struct = generate(&item);

    let out = quote! {
        #[derive(Clone, Debug, PartialEq)]
        #full_struct
    };

    out.into()
}
