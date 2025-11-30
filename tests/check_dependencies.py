#!/usr/bin/env python3
"""Script pour vérifier que toutes les dépendances nécessaires sont installées."""

import sys

REQUIRED_PACKAGES = {
    "pytest": "pytest",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
}

missing_packages = []

print("Vérification des dépendances...")
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}\n")

for module_name, package_name in REQUIRED_PACKAGES.items():
    try:
        __import__(module_name)
        print(f"✅ {package_name} est installé")
    except ImportError:
        print(f"❌ {package_name} est MANQUANT")
        missing_packages.append(package_name)

if missing_packages:
    print(f"\n⚠️  Packages manquants: {', '.join(missing_packages)}")
    print("\nPour installer avec conda:")
    print(f"  conda install {' '.join(missing_packages)} -c conda-forge")
    print("\nOu avec pip:")
    print(f"  pip install {' '.join(missing_packages)}")
    sys.exit(1)
else:
    print("\n✅ Toutes les dépendances sont installées!")
    sys.exit(0)
