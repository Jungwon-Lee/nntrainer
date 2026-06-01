# Test executables for CausalLM

# End-to-end test: Video → VJEPA2 ViT → Projector → LFM2 → Text
if ffmpeg_available
    test_vjepa_lfm2_src = [
        meson.current_source_dir() / 'test_vjepa_lfm2.cpp',
    ]

    test_vjepa_lfm2 = executable('test_vjepa_lfm2',
        test_vjepa_lfm2_src,
        include_directories: [causallm_inc, causallm_inc_minja_inc, causallm_inc_nlohmann_inc],
        dependencies: [nntrainer_dep, nntrainer_ccapi_dep, causallm_layer_dependencies, causallm_dep, ffmpeg_deps],
        build_by_default: true,
        install: false,
    )

    # Module-by-module verification test
    test_verify_modules_src = [
        meson.current_source_dir() / 'test_verify_modules.cpp',
    ]

    test_verify_modules = executable('test_verify_modules',
        test_verify_modules_src,
        include_directories: [causallm_inc, causallm_inc_minja_inc, causallm_inc_nlohmann_inc],
        dependencies: [nntrainer_dep, nntrainer_ccapi_dep, causallm_layer_dependencies, causallm_dep, ffmpeg_deps],
        build_by_default: true,
        install: false,
    )
endif
