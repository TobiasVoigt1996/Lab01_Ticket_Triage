# Lab01 - Support Ticket Triage mit vortrainierten ML-Modellen

## Szenario

Dieses Projekt automatisiert die erste Bearbeitung von Support-Tickets.
Ein eingehendes Ticket wird automatisch kategorisiert, priorisiert und kurz zusammengefasst.

## Ziel

Ziel ist ein Proof of Concept für eine intelligente Prozesskette mit vortrainierten Hugging-Face-Modellen. Es werden keine eigenen Modelle trainiert.

## Prozesskette

1. Laden eines Ticket-Datensatzes
2. Kombination aus `subject` und `body` zum vollständigen Tickettext
3. Klassifikation der Ticket-Kategorie
4. Priorisierung auf Basis der vorhergesagten Kategorie
5. Generierung einer kurzen Zusammenfassung
6. Speicherung der Ergebnisse in `ticket_results.csv`

## Verwendete Technologien

- Python
- pandas
- Hugging Face Transformers
- Zero-Shot Classification
- Text Generation

## Verwendete Modelle

- `facebook/bart-large-mnli` für Zero-Shot-Klassifikation
- `distilgpt2` für einfache Textgenerierung / Zusammenfassung

## Dateien

- `ticket_triage.py`: Hauptskript
- `ticket_dataset.csv`: Eingabedaten
- `ticket_results.csv`: erzeugte Ergebnisse
- `requirements.txt`: benötigte Python-Pakete

## Ausführen

```bash
python ticket_triage.py
pip install -r requirements.txt
python ticket_triage.py


Nutzen
#Proof of Concept zeigt, wie Support-Teams entlastet werden könnten.
#Tickets können automatisch vorsortiert, priorisiert und schneller an die richtige Stelle weitergeleitet werden.


Zu Aufgabenpunkt 3:

Ich habe mein System schrittweise mit 5,10,15,20,25 realen Support-Tickets aus einem Kaggle-Datensatz getestet.
-> Code beinhaltet 25

Die Tickets enthielten unterschiedliche Themen:
- Account-Probleme
- Rechnungsfragen
- technische Störungen
- allgemeine Anfragen
- Netzwerkprobleme


Das Modell konnte die meisten Tickets sinnvoll kategorisieren.
Besonders klare Titel wie "Invoice Details" oder "VPN Access Issue" wurden gut erkannt bzw. könnten an entsprechene Fachkräfte gesandt werden.
Die Kategorisierung funktionierte insgesamt besser als die Priorisierung. Prioritäten sind schwieriger vorherzusagen, weil Dringlichkeit oft subjektiv ist.
Ein technisches Problem kann kritisch oder harmlos sein – das erkennt das Modell nicht immer sicher.

Außerdem waren die automatisch erzeugten Zusammenfassungen teilweise ungenau oder abgeschnitten.
Ein weiteres Limit ist die Sprache: Englische Tickets funktionierten stabiler als deutschsprachige Daten.


Der Proof of Concept zeigt, dass eingehende Tickets automatisch vorsortiert werden können.
Dadurch könnten Support-Teams Zeit sparen:

- Tickets schneller an die richtige Abteilung senden
- dringende Fälle (ggf.!) früher erkennen
- große Ticketmengen effizienter bearbeiten
- Mitarbeiter entlasten

Das System ersetzt (noch) keinen Menschen vollständig, unterstützt aber die Erstbearbeitung sinnvoll.

#Die Zusammenfassungen sind prototypisch und könnten mit einem spezialisierten Summarization-Modell #verbessert werden.
#Für einen produktiven Einsatz wären weitere Tests mit Unternehmensdaten notwendig (Zeit, #Ressourcen).
