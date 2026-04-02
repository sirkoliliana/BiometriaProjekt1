# Biometria - Projekt 1: Przetwarzanie Obrazów
Autorzy: Liwia Jankowska, Liliana Sirko

Aplikacja okienkowa z graficznym interfejsem użytkownika (GUI) do nakładania filtrów i transformacji pikselowych na obrazy. 

## Funkcjonalności
* **Obsługiwane formaty:** `.png`, `.jpg`, `.bmp`.
* **System Pipeline:** Możliwość kolejkowania wielu operacji jedna po drugiej, z funkcją cofania (Undo) i resetowania (Reset All).
* **Wyświetlanie histogramów i projekcji:** Generowanie histogramów dla obrazów oraz projekcji pionowych/poziomych dla obrazów zbinaryzowanych.
* **Transformacje punktowe** 
* **Filtry splotowe**

  
## Struktura projektu
* `app.py` - Główny plik startowy aplikacji.
* `ui/` - Katalog zawierający pliki odpowiedzialne za widoki i interfejs użytkownika.
* `filters.py` - Implementacja filtrów.
* `pixel_transformations.py` - Implementacja algorytmów transformacji na pojedynczych pikselach.
* `pipeline.py` - Logika zarządzania kolejką operacji nakładanych na obraz.
* `state.py` / `models.py` - Zarządzanie stanem aplikacji i strukturami danych.
* `requirements.txt` - Lista zależności wymaganych do uruchomienia projektu.

## Wymagania i instalacja

Aby uruchomić projekt, należy zainstalować wymagane biblioteki:

```bash
pip install -r requirements.txt
