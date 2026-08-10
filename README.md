# DataLens

Interaktive Datenanalyse im Browser – CSV- oder Excel-Datei hochladen, Spalten auswählen, fertige Diagramme erhalten. Gebaut mit [Dash](https://dash.plotly.com/) und [Plotly](https://plotly.com/python/).

## Features

- **Datei-Upload per Drag & Drop** – CSV und Excel (`.xls` / `.xlsx`), automatische Erkennung von kategorischen und numerischen Spalten
- **Fünf Diagrammtypen** – Balkendiagramm, Heatmap, Kreisdiagramm, Boxplot und Streudiagramm, mit automatischen Achsen-Vorschlägen je nach Typ
- **Aggregationen** – Anzahl, Summe, Durchschnitt oder 100 %-gestapelte Prozentwerte
- **Clever sortieren** – Achsen alphabetisch, nach Summe oder per hierarchischem Clustering nach Ähnlichkeit der Verteilungen ordnen
- **Design-Optionen** – sechs Farbpaletten, frei wählbare Schriftgröße, Farben und Hintergrund, Light- und Dark-Mode
- **Export** – Diagramme als SVG herunterladen, aggregierte Daten als Tabelle einsehen

## Installation

```bash
pip install -r requirements.txt
```

## Starten

```bash
python datalens.py
```

Danach im Browser [http://localhost:8050](http://localhost:8050) öffnen.

## Bedienung

1. CSV- oder Excel-Datei in die Upload-Zone ziehen
2. Unter **Datenauswahl** festlegen, welche Spalten gruppiert und welche verglichen werden
3. Unter **Diagramm** den Typ wählen – Achsen und Farbgebung werden automatisch vorgeschlagen und lassen sich anpassen
4. Optional: Sortierung, Beschriftung und Design verfeinern
