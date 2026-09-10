# Test Audio Fixtures

This directory contains test audio fixtures used in the test suite for speech recognition, voice activity detection, and audio processing pipelines.

## `nepali_sample.wav`

- **Source**: [OpenSLR 54 (Large Nepali ASR training data set)](https://www.openslr.org/54/)
- **File Origin**: Utterance `50173ab781.flac` extracted from `asr_nepali_5.zip` and converted to 16kHz mono PCM 16-bit WAV.
- **Duration**: ~3.90 seconds
- **File Size**: ~122 KB (124,878 bytes)
- **Format**: WAV (PCM signed 16-bit little-endian, 16000 Hz, mono)
- **License**: [Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)
- **Copyright**: Copyright 2016, 2017, 2018 Google, Inc.

### Attribution & Citation

```bibtex
@inproceedings{kjartansson-etal-sltu2018,
  title = {{Crowd-Sourced Speech Corpora for Javanese, Sundanese, Sinhala, Nepali, and Bangladeshi Bengali}},
  author = {Oddur Kjartansson and Supheakmungkol Sarin and Knot Pipatsrisawat and Martin Jansche and Linne Ha},
  booktitle = {Proc. The 6th Intl. Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU)},
  year = {2018},
  address = {Gurugram, India},
  month = aug,
  pages = {52--55},
  URL = {http://dx.doi.org/10.21437/SLTU.2018-11}
}
```

## `japanese_sample.wav`

- **Source**: [Mozilla Common Voice (Japanese Corpus)](https://commonvoice.mozilla.org/) via Common Voice 17.0
- **File Origin**: Utterance `common_voice_ja_19499629.mp3` (sentence ID: `15ad6b4189a5cd1670c10c64fdfeb84f01e7d203ccc2b6ebd444961df4b89020`, client ID: `15b7d87a73d28b37664fdf7fea1ff232f89e80ce954c9b90df97ceac3c4e2516428c5a86387beac5e88e2c15c153f3320511ba9bda01152f03fc16666a9121af`) from `test.tsv`, converted to 16kHz mono PCM 16-bit WAV.
- **Duration**: ~3.74 seconds (3.744 seconds)
- **File Size**: ~117 KB (119,886 bytes)
- **Format**: WAV (PCM signed 16-bit little-endian, 16000 Hz, mono)
- **License**: [Creative Commons Zero 1.0 Universal (CC0 1.0 Public Domain Dedication)](https://creativecommons.org/publicdomain/zero/1.0/)
- **Copyright**: Dedicated to the public domain by Mozilla and Common Voice contributors.
- **Ground-Truth Transcript**: `新しい靴をはいて出かけます。`

### Attribution & Citation

```bibtex
@inproceedings{ardila-etal-2020-common,
  title = {{Common Voice: A Massively-Multilingual Speech Corpus}},
  author = {Rosana Ardila and Megan Branson and Kelly Davis and Michael Henretty and Michael Kohler and Josh Meyer and Reuben Morais and Lindsay Saunders and Francis M. Tyers and Gregor Weber},
  booktitle = {Proc. 12th Language Resources and Evaluation Conference (LREC)},
  year = {2020},
  pages = {4218--4222},
  URL = {https://aclanthology.org/2020.lrec-1.520/}
}
```

