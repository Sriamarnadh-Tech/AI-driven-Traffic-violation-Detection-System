import re

def clean_text(text):
    if not text:
        return ""

   
    t = text.upper()

    
    t = re.sub(r"[^A-Z0-9]", "", t)

  
    fixes = {
        
        "O": "0",
        "Q": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
        "|": "1",
        "D": "0",

        
        "8": "B",     
        "0": "O",     
        "1": "I",    
        "5": "S",
        "2": "Z",
        "M": "W",    
        "W": "M"
    }

    corrected = ""
    for ch in t:
        corrected += fixes.get(ch, ch)

  
    corrected_list = list(corrected)

   
    letter_positions = [0,1,4,5]
    digit_positions = [2,3,6,7,8,9]

   
    for i in letter_positions:
        if i < len(corrected_list) and corrected_list[i].isdigit():
            
            swap = {
                "0":"O", "1":"I", "5":"S", "2":"Z",
                "6":"G", "8":"B"
            }
            corrected_list[i] = swap.get(corrected_list[i], corrected_list[i])

    
    for i in digit_positions:
        if i < len(corrected_list) and corrected_list[i].isalpha():
            swap = {
                "O":"0", "I":"1", "L":"1", "Z":"2",
                "S":"5", "B":"8", "G":"6"
            }
            corrected_list[i] = swap.get(corrected_list[i], corrected_list[i])

    final = "".join(corrected_list)

    
    pattern = r"[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{3,4}"
    match = re.findall(pattern, final)

    if match:
        return match[0]

    return final
