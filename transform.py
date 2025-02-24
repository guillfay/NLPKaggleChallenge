import csv

# Nom du fichier d'entrée et du fichier de sortie
input_file = "test_predictions_bert.csv"
output_file = "test_predictions_bert_cleaned.csv"

# Ouverture du fichier d'entrée en lecture et du fichier de sortie en écriture
with (
    open(input_file, "r", encoding="utf-8") as f_in,
    open(output_file, "w", encoding="utf-8", newline="") as f_out,
):
    # On utilise DictReader pour lire le CSV sous forme de dictionnaire
    reader = csv.DictReader(f_in)

    # On définit le nom de la colonne que l'on veut conserver
    fieldnames = ["Predicted_Label"]

    # Création de l'écrivain CSV avec la colonne souhaitée
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()  # On écrit l'en-tête contenant seulement 'Predicted_Label'

    # Pour chaque ligne, on extrait la valeur de 'Predicted_Label' et on écrit dans le fichier de sortie
    for row in reader:
        writer.writerow({"Predicted_Label": row["Predicted_Label"]})
