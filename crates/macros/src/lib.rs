extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse2, parse_macro_input, spanned::Spanned, Attribute, Expr, FieldValue, Fields, Ident,
    ItemStruct, Member, Token, Type, TypePath,
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
                if let Type::Path(TypePath { path: p, .. }) = &mut field.ty {
                    if let Some(segment) = p.segments.first_mut() {
                        segment.ident = Ident::new("SettingsBlock", segment.ident.span());
                        ts_for_ref = quote! {SettingsBlock::from(&value.#name)};
                        ts_for_owned = quote! {SettingsBlock::from(value.#name)};
                        ts_to_ref = quote! {Option::from(&value.#name)};
                        ts_to_owned = quote! {Option::from(value.#name)};
                    }
                }
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
        #[derive(Clone, Debug)]
        #full_struct
    };

    out.into()
}
