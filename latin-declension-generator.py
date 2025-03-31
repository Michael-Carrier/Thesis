import csv
import re

class LatinDeclensionGenerator:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.words = []
        self.load_data()
        
    def load_data(self):
        """Load data from CSV file"""
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                header = next(reader)  # Skip header row
                for row in reader:
                    if len(row) >= 3:
                        latin_word = row[0].strip()
                        category = row[1].strip()
                        definition = row[2].strip()
                        self.words.append({
                            'word': latin_word,
                            'category': category,
                            'definition': definition
                        })
        except Exception as e:
            print(f"Error loading CSV file: {e}")
    
    def save_declensions(self, output_path):
        """Save declensions to a new CSV file"""
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # Create header row with all declension forms
                header = ['latin_word', 'category', 'definition', 
                          'nom_sg', 'gen_sg', 'dat_sg', 'acc_sg', 'abl_sg', 'voc_sg',
                          'nom_pl', 'gen_pl', 'dat_pl', 'acc_pl', 'abl_pl', 'voc_pl']
                writer.writerow(header)
                
                # Process each word
                for word_data in self.words:
                    word = word_data['word']
                    category = word_data['category']
                    definition = word_data['definition']
                    
                    # Generate declensions
                    declensions = self.generate_declensions(word, category)
                    
                    # Create row with all data
                    row = [word, category, definition]
                    row.extend(declensions)
                    writer.writerow(row)
                    
                print(f"Declensions saved to {output_path}")
        except Exception as e:
            print(f"Error saving declensions: {e}")
    
    def generate_declensions(self, word, category):
        """Generate all declension forms based on word and category"""
        # Default to full list of 12 declension forms (sg + pl)
        if "Instructions" in category:  # Verb category
            return self.generate_verb_forms(word)
        elif category == "Time Markers":
            return self.generate_time_marker_forms(word)
        elif category == "Measurement Terms":
            return self.generate_measurement_forms(word)
        else:
            # Default case for unknown categories
            return self.generate_default_forms(word)
            
    def generate_verb_forms(self, word):
        """Generate verb conjugations instead of declensions"""
        # For first conjugation verbs (-are)
        if word.endswith('o') and word[:-1] + 'are' in ['facere', 'ponere', 'servare', 'curare', 'miscere', 'arare', 'serere', 'coquere', 'secare', 'vertere', 'colligere', 'purgare', 'exsiccare']:
            verb_type = self.determine_conjugation(word)
            
            if verb_type == 1:  # First conjugation (-are)
                stem = word[:-1]  # Remove the 'o'
                present_active = [
                    word,                 # 1st person singular
                    f"{stem}as",          # 2nd person singular
                    f"{stem}at",          # 3rd person singular
                    f"{stem}amus",        # 1st person plural
                    f"{stem}atis",        # 2nd person plural
                    f"{stem}ant"          # 3rd person plural
                ]
                
                # For nouns we need 12 forms, so we'll add imperative forms to complete
                imperative = [
                    f"{stem}a",           # singular imperative
                    f"{stem}ate",         # plural imperative
                    f"{stem}are",         # infinitive
                    f"{stem}ans",         # present participle
                    f"{stem}atus",        # perfect passive participle
                    f"{stem}aturus"       # future active participle
                ]
                
                return present_active + imperative
            
            elif verb_type == 2:  # Second conjugation (-ere with long e)
                stem = word[:-1]  # Remove the 'o'
                present_active = [
                    word,                 # 1st person singular
                    f"{stem}es",          # 2nd person singular
                    f"{stem}et",          # 3rd person singular
                    f"{stem}emus",        # 1st person plural
                    f"{stem}etis",        # 2nd person plural
                    f"{stem}ent"          # 3rd person plural
                ]
                
                imperative = [
                    f"{stem}e",           # singular imperative
                    f"{stem}ete",         # plural imperative
                    f"{stem}ēre",         # infinitive
                    f"{stem}ens",         # present participle
                    f"{stem}itus",        # perfect passive participle
                    f"{stem}iturus"       # future active participle
                ]
                
                return present_active + imperative
                
            elif verb_type == 3:  # Third conjugation (-ere with short e)
                stem = word[:-1]  # Remove the 'o'
                present_active = [
                    word,                 # 1st person singular
                    f"{stem}is",          # 2nd person singular
                    f"{stem}it",          # 3rd person singular
                    f"{stem}imus",        # 1st person plural
                    f"{stem}itis",        # 2nd person plural
                    f"{stem}unt"          # 3rd person plural
                ]
                
                imperative = [
                    f"{stem}e",           # singular imperative
                    f"{stem}ite",         # plural imperative
                    f"{stem}ere",         # infinitive
                    f"{stem}ens",         # present participle
                    f"{stem}tus",         # perfect passive participle
                    f"{stem}turus"        # future active participle
                ]
                
                return present_active + imperative
                
            elif verb_type == 4:  # Fourth conjugation (-ire)
                stem = word[:-1]  # Remove the 'o'
                present_active = [
                    word,                 # 1st person singular
                    f"{stem}is",          # 2nd person singular
                    f"{stem}it",          # 3rd person singular
                    f"{stem}imus",        # 1st person plural
                    f"{stem}itis",        # 2nd person plural
                    f"{stem}iunt"         # 3rd person plural
                ]
                
                imperative = [
                    f"{stem}i",           # singular imperative
                    f"{stem}ite",         # plural imperative
                    f"{stem}ire",         # infinitive
                    f"{stem}iens",        # present participle
                    f"{stem}itus",        # perfect passive participle
                    f"{stem}iturus"       # future active participle
                ]
                
                return present_active + imperative
                
            else:
                # Irregular verbs or unknown conjugation
                return [word] * 12
        else:
            # Unknown verb type, return original word
            return [word] * 12
    
    def determine_conjugation(self, verb):
        """Determine the conjugation of a verb based on its infinitive pattern"""
        # Map verbs to their conjugation types
        verb_map = {
            "facio": 3,    # facere (3rd)
            "pono": 3,     # ponere (3rd)
            "servo": 1,    # servare (1st)
            "curo": 1,     # curare (1st)
            "misceo": 2,   # miscere (2nd)
            "aro": 1,      # arare (1st)
            "sero": 3,     # serere (3rd)
            "coquo": 3,    # coquere (3rd)
            "seco": 1,     # secare (1st)
            "verto": 3,    # vertere (3rd)
            "colligo": 3,  # colligere (3rd)
            "purgo": 1,    # purgare (1st)
            "exsicco": 1,  # exsiccare (1st)
        }
        
        return verb_map.get(verb, 0)  # Default to 0 if not found
    
    def generate_time_marker_forms(self, word):
        """Generate declensions for time marker words"""
        declensions = []
        
        # Handle specific time words based on their pattern
        if word == "dies":  # 5th declension, masculine/feminine
            declensions = [
                "dies",      # nom_sg
                "diei",      # gen_sg
                "diei",      # dat_sg
                "diem",      # acc_sg
                "die",       # abl_sg
                "dies",      # voc_sg
                "dies",      # nom_pl
                "dierum",    # gen_pl
                "diebus",    # dat_pl
                "dies",      # acc_pl
                "diebus",    # abl_pl
                "dies"       # voc_pl
            ]
        elif word == "mensis":  # 3rd declension, masculine
            declensions = [
                "mensis",     # nom_sg
                "mensis",     # gen_sg
                "mensi",      # dat_sg
                "mensem",     # acc_sg
                "mense",      # abl_sg
                "mensis",     # voc_sg
                "menses",     # nom_pl
                "mensium",    # gen_pl
                "mensibus",   # dat_pl
                "menses",     # acc_pl
                "mensibus",   # abl_pl
                "menses"      # voc_pl
            ]
        elif word == "tempus":  # 3rd declension, neuter
            declensions = [
                "tempus",     # nom_sg
                "temporis",   # gen_sg
                "tempori",    # dat_sg
                "tempus",     # acc_sg
                "tempore",    # abl_sg
                "tempus",     # voc_sg
                "tempora",    # nom_pl
                "temporum",   # gen_pl
                "temporibus", # dat_pl
                "tempora",    # acc_pl
                "temporibus", # abl_pl
                "tempora"     # voc_pl
            ]
        elif word == "hora":  # 1st declension, feminine
            declensions = [
                "hora",       # nom_sg
                "horae",      # gen_sg
                "horae",      # dat_sg
                "horam",      # acc_sg
                "hora",       # abl_sg
                "hora",       # voc_sg
                "horae",      # nom_pl
                "horarum",    # gen_pl
                "horis",      # dat_pl
                "horas",      # acc_pl
                "horis",      # abl_pl
                "horae"       # voc_pl
            ]
        elif word == "autumnus":  # 2nd declension, masculine
            declensions = [
                "autumnus",     # nom_sg
                "autumni",      # gen_sg
                "autumno",      # dat_sg
                "autumnum",     # acc_sg
                "autumno",      # abl_sg
                "autumne",      # voc_sg
                "autumni",      # nom_pl
                "autumnorum",   # gen_pl
                "autumnis",     # dat_pl
                "autumnos",     # acc_pl
                "autumnis",     # abl_pl
                "autumni"       # voc_pl
            ]
        elif word == "ver":  # 3rd declension, neuter
            declensions = [
                "ver",         # nom_sg
                "veris",       # gen_sg
                "veri",        # dat_sg
                "ver",         # acc_sg
                "vere",        # abl_sg
                "ver",         # voc_sg
                "vera",        # nom_pl (rare)
                "verum",       # gen_pl (rare)
                "veribus",     # dat_pl (rare)
                "vera",        # acc_pl (rare)
                "veribus",     # abl_pl (rare)
                "vera"         # voc_pl (rare)
            ]
        elif word == "aestas":  # 3rd declension, feminine
            declensions = [
                "aestas",      # nom_sg
                "aestatis",    # gen_sg
                "aestati",     # dat_sg
                "aestatem",    # acc_sg
                "aestate",     # abl_sg
                "aestas",      # voc_sg
                "aestates",    # nom_pl
                "aestatum",    # gen_pl
                "aestatibus",  # dat_pl
                "aestates",    # acc_pl
                "aestatibus",  # abl_pl
                "aestates"     # voc_pl
            ]
        elif word == "hiems":  # 3rd declension, feminine
            declensions = [
                "hiems",       # nom_sg
                "hiemis",      # gen_sg
                "hiemi",       # dat_sg
                "hiemem",      # acc_sg
                "hieme",       # abl_sg
                "hiems",       # voc_sg
                "hiemes",      # nom_pl
                "hiemum",      # gen_pl
                "hiemibus",    # dat_pl
                "hiemes",      # acc_pl
                "hiemibus",    # abl_pl
                "hiemes"       # voc_pl
            ]
        elif word == "calendae":  # 1st declension, feminine plural
            # Mostly used in plural form
            declensions = [
                "calendae",     # nom_sg (rare)
                "calendae",     # gen_sg (rare)
                "calendae",     # dat_sg (rare)
                "calendam",     # acc_sg (rare)
                "calenda",      # abl_sg (rare)
                "calendae",     # voc_sg (rare)
                "calendae",     # nom_pl
                "calendarum",   # gen_pl
                "calendis",     # dat_pl
                "calendas",     # acc_pl
                "calendis",     # abl_pl
                "calendae"      # voc_pl
            ]
        elif word == "idus":  # 4th declension, feminine
            declensions = [
                "idus",        # nom_sg
                "idus",        # gen_sg
                "idui",        # dat_sg
                "idum",        # acc_sg
                "idu",         # abl_sg
                "idus",        # voc_sg
                "idus",        # nom_pl
                "iduum",       # gen_pl
                "idibus",      # dat_pl
                "idus",        # acc_pl
                "idibus",      # abl_pl
                "idus"         # voc_pl
            ]
        elif word == "nonae":  # 1st declension, feminine plural
            # Mostly used in plural form
            declensions = [
                "nona",        # nom_sg (rare)
                "nonae",       # gen_sg (rare)
                "nonae",       # dat_sg (rare)
                "nonam",       # acc_sg (rare)
                "nona",        # abl_sg (rare)
                "nona",        # voc_sg (rare)
                "nonae",       # nom_pl
                "nonarum",     # gen_pl
                "nonis",       # dat_pl
                "nonas",       # acc_pl
                "nonis",       # abl_pl
                "nonae"        # voc_pl
            ]
        elif word == "nox":  # 3rd declension, feminine
            declensions = [
                "nox",         # nom_sg
                "noctis",      # gen_sg
                "nocti",       # dat_sg
                "noctem",      # acc_sg
                "nocte",       # abl_sg
                "nox",         # voc_sg
                "noctes",      # nom_pl
                "noctium",     # gen_pl
                "noctibus",    # dat_pl
                "noctes",      # acc_pl
                "noctibus",    # abl_pl
                "noctes"       # voc_pl
            ]
        elif word == "matutinus":  # 1st/2nd declension adjective (masculine form)
            declensions = [
                "matutinus",     # nom_sg masc
                "matutini",      # gen_sg masc
                "matutino",      # dat_sg masc
                "matutinum",     # acc_sg masc
                "matutino",      # abl_sg masc
                "matutine",      # voc_sg masc
                "matutini",      # nom_pl masc
                "matutinorum",   # gen_pl masc
                "matutinis",     # dat_pl masc
                "matutinos",     # acc_pl masc
                "matutinis",     # abl_pl masc
                "matutini"       # voc_pl masc
            ]
        elif word == "vesper":  # 2nd declension, masculine (also sometimes vespera in 1st decl.)
            declensions = [
                "vesper",      # nom_sg
                "vesperi",     # gen_sg
                "vespero",     # dat_sg
                "vesperum",    # acc_sg
                "vespero",     # abl_sg
                "vesper",      # voc_sg
                "vesperi",     # nom_pl
                "vesperorum",  # gen_pl
                "vesperis",    # dat_pl
                "vesperos",    # acc_pl
                "vesperis",    # abl_pl
                "vesperi"      # voc_pl
            ]
        elif word == "meridies":  # 5th declension, masculine
            declensions = [
                "meridies",     # nom_sg
                "meridiei",     # gen_sg
                "meridiei",     # dat_sg
                "meridiem",     # acc_sg
                "meridie",      # abl_sg
                "meridies",     # voc_sg
                "meridies",     # nom_pl (rare)
                "meridierum",   # gen_pl (rare)
                "meridiebus",   # dat_pl (rare)
                "meridies",     # acc_pl (rare)
                "meridiebus",   # abl_pl (rare)
                "meridies"      # voc_pl (rare)
            ]
        else:
            # Default for unknown time markers
            declensions = [word] * 12
        
        return declensions
    
    def generate_measurement_forms(self, word):
        """Generate declensions for measurement terms"""
        declensions = []
        
        # Handle specific measurement words based on their pattern
        if word == "modius":  # 2nd declension, masculine
            declensions = [
                "modius",     # nom_sg
                "modii",      # gen_sg
                "modio",      # dat_sg
                "modium",     # acc_sg
                "modio",      # abl_sg
                "modi",       # voc_sg
                "modii",      # nom_pl
                "modiorum",   # gen_pl
                "modiis",     # dat_pl
                "modios",     # acc_pl
                "modiis",     # abl_pl
                "modii"       # voc_pl
            ]
        elif word == "libra":  # 1st declension, feminine
            declensions = [
                "libra",      # nom_sg
                "librae",     # gen_sg
                "librae",     # dat_sg
                "libram",     # acc_sg
                "libra",      # abl_sg
                "libra",      # voc_sg
                "librae",     # nom_pl
                "librarum",   # gen_pl
                "libris",     # dat_pl
                "libras",     # acc_pl
                "libris",     # abl_pl
                "librae"      # voc_pl
            ]
        elif word == "digitus":  # 2nd declension, masculine
            declensions = [
                "digitus",     # nom_sg
                "digiti",      # gen_sg
                "digito",      # dat_sg
                "digitum",     # acc_sg
                "digito",      # abl_sg
                "digite",      # voc_sg
                "digiti",      # nom_pl
                "digitorum",   # gen_pl
                "digitis",     # dat_pl
                "digitos",     # acc_pl
                "digitis",     # abl_pl
                "digiti"       # voc_pl
            ]
        elif word == "pes":  # 3rd declension, masculine
            declensions = [
                "pes",         # nom_sg
                "pedis",       # gen_sg
                "pedi",        # dat_sg
                "pedem",       # acc_sg
                "pede",        # abl_sg
                "pes",         # voc_sg
                "pedes",       # nom_pl
                "pedum",       # gen_pl
                "pedibus",     # dat_pl
                "pedes",       # acc_pl
                "pedibus",     # abl_pl
                "pedes"        # voc_pl
            ]
        elif word == "sextarius":  # 2nd declension, masculine
            declensions = [
                "sextarius",     # nom_sg
                "sextarii",      # gen_sg
                "sextario",      # dat_sg
                "sextarium",     # acc_sg
                "sextario",      # abl_sg
                "sextarie",      # voc_sg
                "sextarii",      # nom_pl
                "sextariorum",   # gen_pl
                "sextariis",     # dat_pl
                "sextarios",     # acc_pl
                "sextariis",     # abl_pl
                "sextarii"       # voc_pl
            ]
        elif word == "pars":  # 3rd declension, feminine
            declensions = [
                "pars",        # nom_sg
                "partis",      # gen_sg
                "parti",       # dat_sg
                "partem",      # acc_sg
                "parte",       # abl_sg
                "pars",        # voc_sg
                "partes",      # nom_pl
                "partium",     # gen_pl
                "partibus",    # dat_pl
                "partes",      # acc_pl
                "partibus",    # abl_pl
                "partes"       # voc_pl
            ]
        elif word == "pondus":  # 3rd declension, neuter
            declensions = [
                "pondus",      # nom_sg
                "ponderis",    # gen_sg
                "ponderi",     # dat_sg
                "pondus",      # acc_sg
                "pondere",     # abl_sg
                "pondus",      # voc_sg
                "pondera",     # nom_pl
                "ponderum",    # gen_pl
                "ponderibus",  # dat_pl
                "pondera",     # acc_pl
                "ponderibus",  # abl_pl
                "pondera"      # voc_pl
            ]
        else:
            # Default for unknown measurement terms
            declensions = [word] * 12
        
        return declensions
    
    def generate_default_forms(self, word):
        """Generate default forms when category is unknown"""
        # Default to returning the same word 12 times
        return [word] * 12

# Example usage
if __name__ == "__main__":
    # Replace with your actual CSV path
    input_file = "latin_words.csv"
    output_file = "latin_declensions.csv"
    
    generator = LatinDeclensionGenerator(input_file)
    generator.save_declensions(output_file)
    print("Process completed. Check the output file for declensions.")
