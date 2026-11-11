# Motion tracking and camera coverage

This revision adds a per-stone Kalman position/velocity estimator outside the SAM provider.
The raw segmentation observations remain available in `observations.jsonl`. Filtered state
is a separate estimate, not a claim that a hidden stone was observed.

## Camera orientation

The stored sheet basis is unchanged: A button is y=0, B button is y=34.747. Positive x was
defined as image-right in the perspective looking at house A. The previous canvas displayed
that x direction with A nearest the viewer, which mirrored the sheet. The map now transforms
both coordinates for the selected viewing direction. By default it matches the delivery
perspective; users can explicitly look from A or from B. Changing the viewpoint does not
change saved coordinates, masks or physical IDs.

## Coverage and observations

Perspective candidates and masks were previously rejected beyond 12/13 metres from a house,
leaving a gap between cameras. They now cover the sheet's length, with source-bound panel
crops and geometry checks. Image-to-ice measurement variance is approximated from the local
projection Jacobian plus a calibration/body-height uncertainty floor. Distant perspective
measurements consequently have less influence than clear overhead measurements.

Some SAM masks contain the handle alone. A compact foreground granite region beneath the
handle can supply an approximate ground-contact point without modifying the saved SAM mask.
Painted ice is excluded using the calibration background. When that point cannot be recovered,
depth uncertainty is increased instead of treating the handle as a precise ice coordinate.
Low model-quality masks also receive less measurement weight. These are conservative
heuristics, not empirically calibrated probabilities or a general stone detector.

## State estimator

Each axis tracks position and velocity with a 2x2 covariance. Prediction uses a constant
velocity model with acceleration process noise; initial delivery acceleration gets a wider
prior. Measurement correction uses Kalman gain and the Joseph covariance update. The
standard predict/correct equations are described in the
[OpenCV Kalman filter reference](https://docs.opencv.org/4.x/dd/d6a/classcv_1_1KalmanFilter.html).
The small implementation is independent of OpenCV/MLX so it can be tested in the web environment.

- Innovation, maximum-speed and direction gates reject inconsistent measurements.
- New camera association waits for two observations to establish initial velocity.
- Established moving estimates cannot spontaneously step backward along their velocity
  without a recent contact hypothesis. Corrections that hit a bound retain extra uncertainty.
- Motion-gated swept paths can suggest contact. Contact increases velocity uncertainty,
  allowing bumps and reversals rather than forcing an uninterrupted delivery trajectory.
- Resting stones use a time window and hysteresis, preserving identity through occlusion.
- Moving stones are predicted through short gaps. Uncertainty grows; after six seconds
  without measurements, the position stops extrapolating and is explicitly marked lost.
- Per-view acceptance is recorded. A rejected mask is not painted merely because another
  camera has a valid observation of the same stone; rejected memory is removed from SAM.

The replay shows filtered trails, optional raw measurements and approximate 95% covariance
ellipses. Solid markers have measurement support, hollow markers are predicted/held, and
crosses are lost. These ellipses describe the assumed model; they are not measured accuracy
guarantees. The filter is not a complete friction/curl/collision simulator or full 3D model.

## Validation scope

Synthetic tests cover noisy travel in both directions, covariance correction, gap prediction
and recovery, false backwards jumps, low-confidence fragments, and per-view rejection.
Geometry/proposal tests cover both camera orientations, inverted hack direction, ground-point
uncertainty and continued tracking proposals beyond the old thirteen-metre cutoff. A fresh
video run is needed to verify SAM recovery itself; replaying existing masks alone cannot
validate the changed coverage. Broader held-out-recording evaluation remains required.
