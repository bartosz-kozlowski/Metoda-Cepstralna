# 🖼️ Metoda Cepstralna – wygładzanie obrazu w C, OpenMP i CUDA

Projekt przedstawia porównanie wydajności trzech wersji algorytmu **cepstralnego wygładzania obrazu** (ang. *cepstral smoothing*) dla plików JPEG:

- 🔹 Wersja **sekwencyjna (C)**
- 🔸 Wersja **wielowątkowa (OpenMP)**
- ⚡ Wersja **GPU (CUDA)**

Zastosowano **dyskretne przekształcenie cosinusowe (DCT)** do analizy obrazu w dziedzinie częstotliwości i filtracji wysokoczęstotliwościowych składników.

---

## 🎯 Cel projektu

- Porównać czas przetwarzania i jakość wyników trzech wersji algorytmu
- Zbadać wpływ parametru wygładzania `mk` oraz rozdzielczości obrazu

---