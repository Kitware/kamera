"""Multi-sensor rig calibration from a KAMERA calibration flight.

Pipeline: synchronized frames -> COLMAP SfM with INS position priors -> rig bundle
adjustment -> camera models, rig geometry, INS boresight, DIVE homographies, PDF report.
"""
