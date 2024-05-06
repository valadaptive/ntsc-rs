project('ntscrs-ae', ['c', 'cpp'],
  version : '0.1',
  default_options : ['warning_level=3', 'c_std=c99', 'cpp_std=c++14'])

inc = include_directories(
  'sdk/Examples/Headers',
  'sdk/Examples/Headers/SP',
  'sdk/Examples/Resources',
  'sdk/Examples/Util',
)

# Adobe says the following in their SDK guide about compiling PiPL files:
# "If you follow our advice and base your projects on the SDK
# samples, you can simply change the .r file containing your PiPL definition(s), and your plug-
# in's resources will be automagically updated the next time you build. Feel the love. Or, if you
# ever tinker with the custom build steps, feel the pain."
# I have, in fact, felt the pain.

compiler = meson.get_compiler('c')

in_pipl = 'ntscrs-ae-PiPL.r'
if build_machine.system() == 'windows'
  # Process the PiPL resource into a Windows binary resource using the SDK's PiPLtool
  # TODO: Figure out how to decipher this file format and make a better compiler for it (cross-platform builds?)
  # As far as I can tell, these are "Rez" files used waaaaay back in the OSX Carbon API:
  # https://keith.github.io/xcode-man-pages/Rez.1.html
  # In [sdk]/Examples/Resources/AE_General.r, a bunch of AE-specific definitions are used to generate the final resource file
  preprocessed_pipl = compiler.preprocess(in_pipl, include_directories: inc)
  pipl_tool = find_program('sdk/Examples/Resources/PiPLtool')
  gen_pipl = custom_target('pipl_resource', input: preprocessed_pipl[0], output: 'ntscrs-ae-PiPL.rcc', command: [pipl_tool, '@INPUT@', '@OUTPUT@'])

  # Compile that binary resource to include it in the output
  windows = import('windows')
  pipl_resource = windows.compile_resources(gen_pipl)

  # Needs to be defined for building a Windows AE plugin
  add_project_arguments('-D_WINDOWS=1', language : ['c', 'cpp'])
else
  rez = find_program('rez')
  rez_cmd = [rez, '-i', '@SOURCE_ROOT@/sdk/Examples/Headers/', '-i', '@SOURCE_ROOT@/sdk/Examples/Resources/', '@INPUT@', '-o', '@OUTPUT@', '-arch', 'arm64', '-arch', 'x86_64', '-useDF']
  gen_pipl = custom_target('pipl_resource', input: in_pipl, output: 'ntscrs-ae-PiPL.rsrc', command: rez_cmd, build_by_default: true)
endif

if get_option('buildtype').startswith('release')
  cargo_flag = '--release'
  cargo_output = 'release'
else
  cargo_flag = '--debug'
  cargo_output = 'debug'
endif

# Compile the Rust side of things
cbindgen = find_program('cbindgen')
ntscrs_headers = custom_target('ntscrs_headers', input: ['../crates/c-api/src/lib.rs', '../crates/c-api/cbindgen.toml'], output: 'ntscrs.h', command: [cbindgen, '--config', '@INPUT1@', '--output', '@OUTPUT@', '@INPUT0@'])

if build_machine.system() == 'windows'
  c_api_lib = 'ntscrs_c_api.lib'
else
  c_api_lib = 'libntscrs_c_api.a'
endif

# Meson's developers refuse to add the ability for a custom_target output to be in a subdirectory for... reasons?
# Obviously, it's all the other build systems who are wrong for placing their outputs in a subdirectory, and we should complain to them instead,
# like cargo, who has had an --out-dir flag since 2018 that still hasn't been stabilized due to bikeshedding.
pymod = import('python')
python = pymod.find_installation('python3')
cargo_shim_path = join_paths(meson.current_source_dir(), '../cargo-shim.py')

macos_cross = meson.get_external_property('macos-cross', '')

lib_command = [python, cargo_shim_path, 'build', cargo_flag, '-p', 'ntscrs-c-api', '--target-dir=@OUTDIR@', '--out-dir=@OUTDIR@']
if macos_cross == 'intel-to-arm'
  lib_command += '--target=aarch64-apple-darwin'
elif macos_cross == 'arm-to-intel'
  lib_command += '--target=x86_64-apple-darwin'
endif

ntscrs_lib = custom_target('ntscrs_lib', input: [], build_always_stale: true, output: c_api_lib, command: lib_command)

src = files(
  'sdk/Examples/Util/AEFX_SuiteHelper.c',
  'sdk/Examples/Util/AEGP_SuiteHandler.cpp',
  'sdk/Examples/Util/MissingSuiteError.cpp',
  'ntscrs-ae.cpp',
) + ntscrs_headers

if build_machine.system() == 'windows'
  src += pipl_resource

  # Libraries required by Rust
  ws2_32 = compiler.find_library('ws2_32')
  userenv = compiler.find_library('userenv')
  ntdll = compiler.find_library('ntdll')
  bcrypt = compiler.find_library('bcrypt')

  lib_deps = [ws2_32, userenv, ntdll, bcrypt]
else
  lib_deps = []
endif

if macos_cross == 'intel-to-arm'
  add_project_arguments(['-arch', 'arm64'], language: ['c', 'cpp'])
  add_project_link_arguments(['-arch', 'arm64'], language: ['c', 'cpp'])
elif macos_cross == 'arm-to-intel'
  add_project_arguments(['-arch', 'x86_64'], language: ['c', 'cpp'])
  add_project_link_arguments(['-arch', 'x86_64'], language: ['c', 'cpp'])
endif

lib = shared_library('ntsc-rs-ae', sources: src,
  install : true, include_directories : inc, dependencies: lib_deps, link_with: [ntscrs_lib], name_suffix: 'aex', install_dir: '.')
