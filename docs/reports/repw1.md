# Report week 1

[TOC]

## Project summary

This project focuses on the classification of electroencephalograms (EEG) from healthy children (CN) versus children with Developmental Dyslexia (DD). The patients are exposed to a prolonged white noise stimulus at specific frequencies (4 Hz, 16 Hz, and 40 Hz) for 2.5 minutes each, in both ascending and descending order, with each test lasting a total of 2.5 × 6 minutes. During this period, brain activity is recorded using a 32-channel EEG (31 + control). The study aims to analyze a specific frequency band from each EEG channel, either the theta or gamma band.

To properly interpret the results of the proposed methodology in the context of DD, the Temporal Sampling Framework (TSF) will be applied. This theory suggests that changes in temporal sampling at slow time scales in an auditory signal can lead to difficulties in language and music development. These changes may be linked to differences in how auditory information is received, potentially arising from variations in the ability to distinguish sound variations and the information perceived about them. TSF posits that atypical oscillatory sampling at different temporal rhythms may be the cause of phonological impairments in DD.

It is hypothesized that a healthy brain will adapt to the stimulus, causing brain activity in regions associated with phonetic aspects of speech to adjust to the stimulus, thereby attenuating the signal recorded by the electrode.

To characterize the temporal windows based on the chosen exposure frequency, Recurrence Quantification Analysis (RQA) from chaos theory will be employed. The approach involves taking the EEG signal from a specific channel and a selected frequency band (theta or gamma) and transforming the original time series using a chaotic parameter (RQE) that more effectively highlights changes in signal descriptors (mean, variance, etc.). Samples where this change is detected will correspond to brains exhibiting attractors in phase space (healthy), while samples without significant changes will correspond to more deterministic subjects, possibly those with DD.

## Objectives for this week

- [ ] Analyze RQA paper [Orlando, G., & Zimatore, G. (2018). RQA correlations on real business cycles time series. SSRN.](https://d1wqtxts1xzle7.cloudfront.net/70687546/0035-0041-libre.pdf?1636043587=&response-content-disposition=inline%3B+filename%3DRQA_correlations_on_real_business_cycles.pdf&Expires=1741034001&Signature=JjT~qvESm9uqVV2j-EEMqEvIa~wGZxN71NPq8laB2uUZUGfJdjj~J7XsR5j2MVdeDRaDCbaLZ3YUS~2TzKEFreurwGI2H~fdaqpPO-6g~m47ggITqVO~L-v1toLR4BHeZbi0vWskVRvO8yH1hAJn~XtmPsGZhDu5HqYMWxAB6ynScWboyLoZVS5PZU8yPYPXTx45m9r5Kvm7sjf95YqIvckKu0Cj9rIUIt5ggSuUcXgzSdTji-osrHnJ-pE7IkTEAg~UuHhsNEYN8eqwwsNJ3eotpjbl-oozxY7lL6GuNFBJhysXz7nxxytf1iXKgRXnbnsjRNjvBXoiEa8hBSKAnQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA):
  - [ ] Generate artifical data to simulate a signal whose mean and variance change in time: $\phi(t) \space / \space \mu_{\phi}(t), \space \sigma_{\phi}(t)$.
  - [ ] Code the RQE metric to get a similar result as the paper
- [ ] Write a summary about the workflow intended for the project.

## Weekly Project Time Tracking

| Date       | Task Description            | Hours Spent | Category/Type | Notes |
|------------|----------------------------|------------|--------------|-------|
| 2025-03-03 | First meeting with Ignacio      | 2.5        | Meetings | The basis of the project where settled, including the input data and methodology |
| 2025-03-03 | Bibliography review                | 1.0        | Research              |       |

