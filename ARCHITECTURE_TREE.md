# SnapTalk Final Architecture Tree

This document separates the current active pipeline modules from obsolete modules kept only for cleanup visibility.

## Active Files

snaptalk/
  app/
    main.py
    core/
      config.py
    routers/
      pipeline_vlm.py
      translation.py
      tts.py
      pronunciation.py
    schemas/
      pipeline.py
      translation.py
      speech.py
      vision.py
    services/
      detection/
        __init__.py
        snap_learn_vlm.py
      segmentation/
        __init__.py
        service.py
      recognition/
        __init__.py
        vlm_providers/
          __init__.py
          base.py
          qwen2vl.py
          openai_gpt4v.py
      translation/
        __init__.py
        service.py
      tts/
        __init__.py
        service.py
      pronunciation/
        __init__.py
        service.py
        pronunciation_lab.py
    utils/
      network_security.py
  scripts/
    snap_learn.py
    pronunciation_lab.py
  data/
    seed_translations.json
    yolo_world_vocab.txt
  requirements.txt
  README.md

## Removed as Obsolete in this cleanup

snaptalk/
  app/
    services/
      translation_service.py  # replaced by app/services/translation/service.py
      tts_service.py          # replaced by app/services/tts/service.py
      vlm_experiment/
      snap_learn_service.py
      vision_pipeline.py
      vision_local.py
    routers/
      pipeline.py
      vision.py
