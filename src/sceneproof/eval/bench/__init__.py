"""Research benchmark: hand-labelled person boxes from public datasets -> zone-event ground truth.

The ground-truth events are derived, not drawn: the datasets' hand-made person boxes are fed through the
SAME ZoneMonitor (feet point, entry/exit confirmation times, dwell threshold, lost-track grace) that the
product runs on detector tracks. The only difference between the ground truth and a prediction is
therefore who produced the boxes — a person (dataset annotators) or the detector + tracker.

Modules: data (sources, manifest, box loaders), episodes (GT/prediction episodes), match (spatio-temporal
matching), split (frozen train/dev/test assignment), run (CLI with the one-shot test lock).
"""
