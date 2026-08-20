"""Traducciones de las clases originales del modelo, sin modificar model.names."""

LABELS_ES = {
    "artery_forceps": "Pinza hemostática",
    "aspirator": "Aspirador",
    "bending_shear": "Tijera curva",
    "circular_spoon": "Cucharilla circular",
    "core_needle": "Aguja de biopsia",
    "fine_needle": "Aguja fina",
    "iris_scissors": "Tijera de iris",
    "operating_scissors": "Tijera quirúrgica",
    "rongeur_forceps_1": "Pinza gubia tipo 1",
    "rongeur_forceps_2": "Pinza gubia tipo 2",
    "scalpel": "Bisturí",
    "stripping": "Instrumento de separación",
    "tweezers": "Pinzas",
    "wire_grabbing_pliers": "Alicate para alambre",
}

INSTRUMENT_DESCRIPTIONS_ES = {
    "artery_forceps": "Pinza utilizada para sujetar vasos y controlar el sangrado.",
    "aspirator": "Instrumento que retira líquidos del campo quirúrgico mediante succión.",
    "bending_shear": "Tijera curva para cortar tejidos o materiales quirúrgicos.",
    "circular_spoon": "Cucharilla empleada para retirar o manipular pequeñas porciones de tejido.",
    "core_needle": "Aguja diseñada para obtener muestras cilíndricas de tejido.",
    "fine_needle": "Aguja delgada utilizada para punciones y toma de muestras pequeñas.",
    "iris_scissors": "Tijera pequeña y precisa para cortes delicados.",
    "operating_scissors": "Tijera de uso general para cortar tejido o material quirúrgico.",
    "rongeur_forceps_1": "Pinza robusta tipo 1 para retirar pequeños fragmentos de tejido o hueso.",
    "rongeur_forceps_2": "Pinza robusta tipo 2 para retirar pequeños fragmentos de tejido o hueso.",
    "scalpel": "Instrumento de hoja afilada utilizado para realizar incisiones.",
    "stripping": "Instrumento empleado para separar o desprender tejido.",
    "tweezers": "Pinza fina para sujetar y manipular tejidos u objetos pequeños.",
    "wire_grabbing_pliers": "Alicate diseñado para sujetar y manipular alambre quirúrgico.",
}


def get_spanish_label(class_name: str) -> str:
    return LABELS_ES.get(class_name, class_name.replace("_", " ").capitalize())
