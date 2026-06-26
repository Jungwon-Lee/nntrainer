smallthinker_src = [
    meson.current_source_dir() / 'smallthinker_causallm.cpp',
]

smallthinker_inc = include_directories('.')

causallm_smallthinker_moe_layer_src = [
    meson.current_source_dir() / 'smallthinker_moe_layer.cpp',
    meson.current_source_dir() / 'smallthinker_moe_layer_slim.cpp',

]

causallm_smallthinker_moe_layer = shared_library(
    'smallthinker_moe_layer',
    causallm_smallthinker_moe_layer_src,
    include_directories: [causallm_layer_inc, smallthinker_inc],
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_smallthinker_moe_layer_dep = declare_dependency(
    link_with: causallm_smallthinker_moe_layer,
    include_directories: smallthinker_inc
)

causallm_src += smallthinker_src
causallm_inc += smallthinker_inc
causallm_layer_dependencies += [causallm_smallthinker_moe_layer_dep]
