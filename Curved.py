import cv2
import numpy as np

# Define region of interest (ROI)
ROI = (170, 140, 300, 300)  # (x, y, width, height)


# Function to interpolate curves using polynomial fit and calculate the midpoint
def interpolate_curves(curve1, curve2, poly_deg=10, n_points=200):
    def fit_curve(curve):
        x, y = curve[:, 0], curve[:, 1]
        # Ensure x is strictly increasing
        sorted_indices = np.argsort(x)
        x_sorted, y_sorted = x[sorted_indices], y[sorted_indices]
        # Fit a polynomial curve (degree 3)
        poly_coefs = np.polyfit(x_sorted, y_sorted, poly_deg)
        # Generate new x values for interpolation
        new_x = np.linspace(x_sorted.min(), x_sorted.max(), n_points)
        new_y = np.polyval(poly_coefs, new_x)
        return np.column_stack((new_x, new_y)).astype(int)

    new_curve1 = fit_curve(curve1)
    new_curve2 = fit_curve(curve2)

    # Midpoint between the two curves
    midpoints = np.mean([new_curve1, new_curve2], axis=0).astype(int)
    return new_curve1, new_curve2, midpoints


# Function to detect curves and apply ROI
def detect_curves(frame):
    x, y, w, h = ROI
    roi = frame[y:y + h, x:x + w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (9, 9), 0), 70, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 2:
        return None, None, None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Flatten contours to 2D array
    curve1 = contours[0].reshape(-1, 2)
    curve2 = contours[1].reshape(-1, 2)

    # Fit curves with polynomial interpolation
    curve1, curve2, midpoints = interpolate_curves(curve1, curve2)

    return curve1 + [x, y], curve2 + [x, y], midpoints + [x, y]


# Main loop
def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curve1, curve2, midpoints = detect_curves(frame)

        if curve1 is not None:
            # Draw curves and midpoints
            for point in curve1:
                cv2.circle(frame, tuple(point), 1, (0, 0, 255), -1)
            for point in curve2:
                cv2.circle(frame, tuple(point), 1, (0, 255, 0), -1)
            for i in range(len(midpoints) - 1):
                cv2.line(frame, tuple(midpoints[i]), tuple(midpoints[i + 1]), (255, 0, 0), 2)

        # Draw the ROI rectangle
        cv2.rectangle(frame, (ROI[0], ROI[1]), (ROI[0] + ROI[2], ROI[1] + ROI[3]), (255, 255, 0), 2)
        cv2.imshow("Curve Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
