from shapely.geometry import box, Point
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2
from matplotlib.patches import Polygon as MatplotlibPolygon

# Définir les coordonnées des points du polygone de la ROI
points = [(0, 200), (330, 95), (85, 140), (75, 110), (360, 60), (385, 100), (240, 478), (0, 478)]

roi_polygon = Polygon(points)

# Charger l'image
image = cv2.imread('A25.jpg')

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Créer un polygone Matplotlib pour représenter la ROI
roi_matplotlib_polygon = MatplotlibPolygon(points, fill=False, edgecolor='red')

# Afficher la ROI sur l'image
ax = plt.gca()
ax.add_patch(roi_matplotlib_polygon)
plt.show()

# Vérifier si un point est à l'intérieur de la ROI (exemple)
point_a_verifier = Point(350, 250)
est_a_l_interieur = roi_polygon.contains(point_a_verifier)

if est_a_l_interieur[0] == True:
    print("aaaaaaaaaaaa")

# # Extraire la région d'intérêt (ROI) (exemple)
# roi = image[y1:y2, x1:x2]

# # Sauvegarder la région d'intérêt dans un fichier image
# cv2.imwrite('roi.jpg', roi)
