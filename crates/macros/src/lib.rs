extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    AngleBracketedGenericArguments, Attribute, Expr, FieldValue, Fields, FnArg, GenericArgument,
    Ident, ItemStruct, Member, Pat, PatType, PathArguments, PathSegment, Stmt, Token, Type,
    TypePath, parse::Parse, parse::ParseStream, parse_macro_input, parse2, spanned::Spanned,
};

fn replace_type_param_in_generic_args(
    args: &mut AngleBracketedGenericArguments,
    from: &str,
    to: &TokenStream2,
) {
    for arg in &mut args.args {
        if let GenericArgument::Type(Type::Path(type_path)) = arg {
            if type_path.path.segments.len() == 1 {
                let segment = &type_path.path.segments[0];
                if segment.ident == from {
                    // Replace with the new type
                    if let Ok(new_type) = syn::parse2::<Type>(to.clone()) {
                        *arg = GenericArgument::Type(new_type);
                    }
                }
            }
        }
    }
}

fn replace_ident_in_stmt(stmt: &Stmt, from: &str, to: &TokenStream2) -> Stmt {
    let mut stmt = stmt.clone();

    // Assuming the body is a single expression statement (method call or function call)
    if let Stmt::Expr(expr, _) = &mut stmt {
        if let Expr::Call(call) = expr {
            // Handle function calls like `foo::<S>(...)`
            if let Expr::Path(path_expr) = &mut *call.func {
                for segment in &mut path_expr.path.segments {
                    if let PathArguments::AngleBracketed(args) = &mut segment.arguments {
                        replace_type_param_in_generic_args(args, from, to);
                    }
                }
            }
        }
    }

    stmt
}

/// Parse a function with its body for SIMD dispatch
struct SimdDispatchInput {
    attrs: Vec<Attribute>,
    vis: syn::Visibility,
    sig: syn::Signature,
    body: Vec<Stmt>,
}

impl Parse for SimdDispatchInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis = input.parse()?;
        let sig = input.parse()?;
        let content;
        syn::braced!(content in input);
        let body = content.call(syn::Block::parse_within)?;

        Ok(SimdDispatchInput {
            attrs,
            vis,
            sig,
            body,
        })
    }
}

/// A procedural macro that generates SIMD-dispatched versions of a function.
///
/// This macro takes a function that calls an "inner" function and generates specialized
/// versions for different SIMD architectures (SSE4.1, AVX2, NEON, WASM).
///
/// The macro expects an attribute specifying the SIMD type parameter name (e.g., `S`).
/// This identifier will be replaced with the appropriate SIMD type in each specialized function.
///
/// The generated function returns `Option<T>` where `T` is the original return type.
/// It returns `Some(result)` when SIMD support is available and used, or `None` when
/// no SIMD support is available.
///
/// # Example
///
/// ```ignore
/// #[simd_dispatch(S)]
/// fn filter_signal_simd<const ROWS: usize>(
///     &self,
///     signal: &mut [&mut [f32]; ROWS],
///     initial: [f32; ROWS],
///     scale: f32,
///     delay: usize,
/// ) {
///     self.filter_signal_in_place_fixed_size_simdx4::<S, ROWS>(
///         signal, initial, scale, delay
///     );
/// }
/// ```
///
/// Then you can use it like:
/// ```ignore
/// if self.filter_signal_simd(signal, initial, scale, delay).is_none() {
///     // Fallback to non-SIMD version
///     self.filter_signal_in_place_fixed_size::<4, ROWS>(signal, initial, scale, delay);
/// }
/// ```
///
/// This will generate dispatch functions where `S` is replaced with `SseF32x4`, `AvxF32x4`,
/// `ArmF32x4`, or `WasmF32x4` depending on the target architecture.
#[proc_macro_attribute]
pub fn simd_dispatch(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as SimdDispatchInput);

    // Parse the SIMD type parameter name from the attribute
    let simd_param_name = if attr.is_empty() {
        "S".to_string()
    } else {
        let simd_ident = parse_macro_input!(attr as Ident);
        simd_ident.to_string()
    };

    let attrs = &input.attrs;
    let vis = &input.vis;
    let sig = &input.sig;
    let fn_name = &sig.ident;
    let generics = &sig.generics;
    let inputs = &sig.inputs;
    let body = &input.body;

    // Extract the original return type and wrap it in Option
    let original_output = &sig.output;
    let return_type = match original_output {
        syn::ReturnType::Default => quote! { () },
        syn::ReturnType::Type(_, ty) => quote! { #ty },
    };
    let option_output = quote! { -> Option<#return_type> };

    // Build generic parameters for the specialized functions
    let generic_params = if generics.params.is_empty() {
        quote! {}
    } else {
        let params = &generics.params;
        quote! { < #params > }
    };

    // Build the where clause if it exists
    let where_clause = &generics.where_clause;

    // Extract parameter names for forwarding
    let param_names: Vec<_> = inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(PatType { pat, .. }) = arg {
                if let Pat::Ident(ident) = &**pat {
                    return Some(&ident.ident);
                }
            }
            None
        })
        .collect();

    // Replace the SIMD type parameter with specific types for each architecture
    let sse41_body: Vec<_> = body
        .iter()
        .map(|stmt| replace_ident_in_stmt(stmt, &simd_param_name, &quote! { SseF32x4 }))
        .collect();

    let avx2_body: Vec<_> = body
        .iter()
        .map(|stmt| replace_ident_in_stmt(stmt, &simd_param_name, &quote! { AvxF32x4 }))
        .collect();

    let neon_body: Vec<_> = body
        .iter()
        .map(|stmt| replace_ident_in_stmt(stmt, &simd_param_name, &quote! { ArmF32x4 }))
        .collect();

    let wasm_body: Vec<_> = body
        .iter()
        .map(|stmt| replace_ident_in_stmt(stmt, &simd_param_name, &quote! { WasmF32x4 }))
        .collect();

    // Generate SSE4.1 dispatch function (x86_64 only)
    let sse41_fn_name = Ident::new(&format!("{}_dispatch_sse41", fn_name), fn_name.span());
    let sse41_dispatch = quote! {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse4.1")]
        #(#attrs)*
        unsafe fn #sse41_fn_name #generic_params ( #inputs ) #original_output #where_clause {
            use crate::f32x4::x86_64::SseF32x4;
            unsafe {
                #(#sse41_body)*
            }
        }
    };

    // Generate AVX2 dispatch function (x86_64 only)
    let avx2_fn_name = Ident::new(&format!("{}_dispatch_avx2", fn_name), fn_name.span());
    let avx2_dispatch = quote! {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2", enable = "fma")]
        #(#attrs)*
        unsafe fn #avx2_fn_name #generic_params ( #inputs ) #original_output #where_clause {
            use crate::f32x4::x86_64::AvxF32x4;
            unsafe {
                #(#avx2_body)*
            }
        }
    };

    // Generate NEON dispatch function (aarch64 only)
    let neon_fn_name = Ident::new(&format!("{}_dispatch_neon", fn_name), fn_name.span());
    let neon_dispatch = quote! {
        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #(#attrs)*
        unsafe fn #neon_fn_name #generic_params ( #inputs ) #original_output #where_clause {
            use crate::f32x4::aarch64::ArmF32x4;
            unsafe {
                #(#neon_body)*
            }
        }
    };

    // Generate WASM dispatch function (wasm32 with simd128 only)
    let wasm_fn_name = Ident::new(&format!("{}_dispatch_wasm", fn_name), fn_name.span());
    let wasm_dispatch = quote! {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        #(#attrs)*
        unsafe fn #wasm_fn_name #generic_params ( #inputs ) #original_output #where_clause {
            use crate::f32x4::wasm32::WasmF32x4;
            unsafe {
                #(#wasm_body)*
            }
        }
    };

    // Generate the main dispatch function that returns Option
    let main_dispatch = quote! {
        #(#attrs)*
        #[inline(always)]
        #vis fn #fn_name #generic_params ( #inputs ) #option_output #where_clause {
            use crate::f32x4::{SupportedSimdType, get_supported_simd_type};

            match get_supported_simd_type() {
                #[cfg(target_arch = "x86_64")]
                SupportedSimdType::Sse41 => unsafe {
                    Some(#sse41_fn_name( #(#param_names),* ))
                },
                #[cfg(target_arch = "x86_64")]
                SupportedSimdType::Avx2 => unsafe {
                    Some(#avx2_fn_name( #(#param_names),* ))
                },
                #[cfg(target_arch = "aarch64")]
                SupportedSimdType::Neon => unsafe {
                    Some(#neon_fn_name( #(#param_names),* ))
                },
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                SupportedSimdType::Wasm => unsafe {
                    Some(#wasm_fn_name( #(#param_names),* ))
                },
                _ => None
            }
        }
    };

    // Combine all generated code
    let output = quote! {
        #sse41_dispatch
        #avx2_dispatch
        #neon_dispatch
        #wasm_dispatch
        #main_dispatch
    };

    output.into()
}

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
