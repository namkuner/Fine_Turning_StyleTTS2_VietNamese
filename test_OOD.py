
with open("Data/OOD_data.txt", 'r', encoding='utf-8') as f:
    tl = f.readlines()


_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔ|ǁǂǃˈˌːˑʼ'̩'+"
_other = '@=`-_~*#01234567'


symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)  + list(_other)

with open("Data/OOD_data.txt", 'r', encoding='utf-8') as f:
    tl = f.readlines()
    for i in tl:
        print(len(i))


