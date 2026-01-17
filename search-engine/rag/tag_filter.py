import re
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Optional

# ===========================
# 1) PATH & LOAD KNOWLEDGE
# ===========================

BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "chemical_tags_from_kb.json"

with open(KB_PATH, "r", encoding="utf-8") as f:
    CHEMICAL_KB = json.load(f)


# ===========================
# 2) NORMALIZATION UTILITIES
# ===========================

_space_re = re.compile(r"\s+")
_non_alnum_re = re.compile(r"[^a-z0-9\s]+")

def normalize(text: str) -> str:
    if not text:
        return ""

    text = text.lower().strip()
    text = text.replace("đ", "d")

    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    # loại bỏ toàn bộ dấu câu và ký tự đặc biệt
    text = _non_alnum_re.sub(" ", text)

    text = _space_re.sub(" ", text)
    return text.strip()


# Chuẩn hóa sẵn dữ liệu trong KB
for k, v in CHEMICAL_KB.items():
    v["crops"] = [normalize(x) for x in v.get("crops", [])]
    v["diseases"] = [normalize(x) for x in v.get("diseases", [])]
    v["pests"] = [normalize(x) for x in v.get("pests", [])]
    v["weeds"] = [normalize(x) for x in v.get("weeds", [])]


# ===========================
# 3) ALIASES (BẠN TỰ COPY ĐẦY ĐỦ SAU)
# ===========================

CHEMICAL_ALIASES = {
    "24-epi-brassinolide": ["brassinolid 24-epi", "hooc mon brassinolide 24-epi", "brassinolide"],
    "24-epibrassinolide": ["brassinolid 24-epi", "hooc mon brassinolide 24-epi", "24-epibrassinolide"],
    "abamectin": ["abamectin", "thuoc tru sau abamectin"],
    "chlormequat": ["chat dieu hoa sinh truong chlormequat", "chlormequat"],
    "diuron": ["diuron", "thuoc diet co diuron"],
    "fomesafen": ["fomesafen", "thuoc diet co fomesafen"],
    "fosthiazate": ["fosthiazate", "thuoc diet sau fosthiazate"],
    "paclobutrazol": ["chat dieu hoa sinh truong paclobutrazol", "thuoc dieu hoa sinh truong paclobutrazol", "paclobutrazol"],
    "profenofos": ["profenofos", "thuoc tru sau profenofos"],
    "spirotetramat": ["spirotetramat", "thuoc tru sau spirotetramat"],
    "sulfur": ["chat diet nam sulfur", "luu huynh", "sulfur"],
    "abinsec-oxatin-1-8ec": ["abinsec oxatin 1 8ec", "thuoc abinsec oxatin 1 8ec"],
    "abound": ["abound", "thuoc diet nam abound"],
    "acetamiprid": ["acetamiprid", "thuoc tru sau acetamiprid"],
    "acetochlor": ["acetoclor", "acetochlor"],
    "acxonik": ["acxonik"],
    "adorn": ["adorn"],
    "aflatoxin": ["aflatoxin"],
    "aflatoxins": ["aflatoxin"],
    "agar": ["agar"],
    "agrohigh": ["agrohigh"],
    "aliette": ["aliette"],
    "alpha-cypermethrin": ["alpha cypermethrin", "alpha xi permethrin", "permethrin", "cypermethrin", "alpha-cypermethrin"],
    "ametryn": ["ametrin", "ametryn"],
    "amino": ["amino"],
    "amisulbrom": ["amisulbrom"],
    "amonium-nitrate": ["amonium nitrat", "phot pho amoni"],
    "antibiotic": ["khang sinh"],
    "antibiotics": ["khang sinh"],
    "atrazine": ["atrazin", "atrazine"],
    "avermectin": ["avermectin"],
    "avermectin-b1a": ["avermectin b1a", "avermectin"],
    "avermectin-b1b": ["avermectin b1b", "avermectin"],
    "azol-450sc": ["azol 450sc"],
    "azoxystrobin": ["azoxistrobin", "azoxystrobin"],
    "azoxystrobincrop:rice": ["azoxystrobin lua"],
    "azoxytrobin": ["azoxistrobin", "azoxytrobin"],
    "bacillus-subtilis": ["bacillus subtilis", "bacillus-subtilis", "bacillus", "subtilis"],
    "badge-x2": ["badge x2"],
    "bentazone": ["bentazon", "bentazone"],
    "beta-cypermethrin": ["beta cypermethrin", "beta-cypermethrin", "cypermethrin", "permethrin"],
    "bifenazate": ["bifenazat", "bifenazate"],
    "bifenthrin": ["bi fen trin", "bifenthrin"],
    "bismerthiazol": ["bismerthiazol"],
    "bismerthiazole": ["bismerthiazol", "bismerthiazole"],
    "boscalid": ["boscalid"],
    "brodifacoum": ["brodifacoum"],
    "buprofezin": ["bu profezin", "buprofezin"],
    "c10-alcohol-ethoxylate": ["c10 alcohol ethoxylate", "alcohol", "ethoxylate", "c10"],
    "cabona": ["thuoc diet sau cabona", "thuoc tru sau cabona"],
    "calcium-hydroxide": ["bot canxi", "canxi hydroxit", "thuoc tri benh canxi"],
    "calcium-nitrate": ["canxi nitrat", "phan bon canxi nitrat"],
    "cartap-hydrochloride": ["cartap hydroclorua", "thuoc diet sau cartap", "cartap", "hydrochloride", "cartap-hydrochloride"],
    "chloramphenicol": ["cloramphenicol", "thuoc khang sinh cloramphenicol"],
    "chlorfenapyr": ["clorfenapyr", "thuoc diet sau clorfenapyr", "chlorfenapyr"],
    "chlorothalonil": ["clorotalonil", "thuoc tri nam clorotalonil", "chlorothalonil"],
    "chlorothalonil-50sc": ["clorotalonil 50sc", "thuoc tri nam clorotalonil 50sc", "chlorothalonil"],
    "chlorothalonil-75wp": ["clorotalonil 75wp", "thuoc tri nam clorotalonil 75wp", "chlorothalonil"],
    "chlorpyrifos-ethyl": ["clorpyrifos etyl", "thuoc diet sau clorpyrifos etyl", "chlorpyrifos"],
    "chlorpyrifos-methyl": ["clorpyrifos metyl", "thuoc diet sau clorpyrifos metyl", "chlorpyrifos", "chlorpyrifos-methyl"],
    "contact-fungicide": ["thuoc diet nam tiep xuc", "thuoc tri nam tiep xuc"],
    "copper-based-fungicide": ["thuoc diet nam goc dong", "thuoc tri nam goc dong"],
    "copper-based-pesticide": ["thuoc diet sau goc dong", "thuoc tru sau goc dong"],
    "copper-chelate": ["dong chelate", "dong hoa hop chelate"],
    "copper-fungicide": ["thuoc diet nam dong", "thuoc tri nam dong"],
    "copper-hydroxide": ["dong bot hydroxit", "dong hydroxit"],
    "copper-ii-hydroxide": ["dong hydroxit", "dong ii hydroxit"],
    "copper-ion": ["dong ion", "ion dong"],
    "copper-oxychloride": ["dong oxyclorid", "dong oxyclorua"],
    "copper-sulfate": ["dong sulfat", "dong sunfat"],
    "cuprous-oxide": ["dong i oxit", "dong oxit"],
    "cyazofamid": ["cyazofamid", "thuoc diet nam cyazofamid"],
    "cyclopiazonic-acid": ["axit cyclopiazonic", "chat doc cyclopiazonic"],
    "cymoxanil": ["cymoxanil", "thuoc diet nam cymoxanil"],
    "cypermethrin": ["cypermethrin", "thuoc diet sau cypermethrin", "thuoc tru sau cypermethrin"],
    "cyromazine": ["cyromazine", "thuoc diet sau cyromazine"],
    "daconil": ["daconil", "thuoc tri nam daconil"],
    "dcpa": ["dcpa", "thuoc diet co dcpa"],
    "dieu-hoa-sinh-truong": ["dieu hoa sinh truong", "dieu hoa", "sinh truong"],
    "decco-salt-no-19": ["decco muoi so 19", "phan bon decco muoi 19"],
    "deltamethrin": ["deltamethrin", "thuoc diet sau deltamethrin"],
    "deoxynivalenol": ["deoxynivalenol", "doc khuan deoxynivalenol"],
    "dextrose": ["duong dextrose", "duong gluco"],
    "diafenthiuron": ["diafenthiuron", "thuoc diet sau diafenthiuron"],
    "dichloran": ["dichloran", "thuoc diet nam dichloran"],
    "difenoconazole": ["difenoconazole", "thuoc tri nam difenoconazole"],
    "dimethoate": ["dimethoate", "thuoc diet sau dimethoate"],
    "dimethomorph": ["dimethomorph", "thuoc tri nam dimethomorph"],
    "dinotefuran": ["thuoc diet sau dinotefuran", "thuoc tru sau dinotefuran", "dinotefuran"],
    "diquat-dibromide": ["thuoc diet co diquat", "thuoc diet co diquat dibromide", "diquat-dibromide"],
    "disease-control-chemical": ["thuoc diet benh", "thuoc phong tru benh"],
    "disease-fungicide-group-a": ["thuoc diet nam nhom a"],
    "disinfectant": ["thuoc khu trung", "thuoc tay trung"],
    "dithianon": ["thuoc diet nam dithianon", "dithianon"],
    "emamectin-benzoate": ["thuoc diet sau emamectin benzoate", "thuoc tru sau emamectin benzoate", "emamectin", "benzoate", "emamectin-benzoate"],
    "enable": ["thuoc diet sau enable"],
    "ethanol-70": ["con 70 do", "cuu am 70 do", "ethanol 70 do"],
    "ethephon": ["thuoc kich thich ethephon", "thuoc kich thich ra hoa ethephon", "ethephon"],
    "ethyl-alcohol": ["con etylic", "cuu am etylic"],
    "ethyl-alcohol-70": ["con etylic 70 do", "cuu am etylic 70 do"],
    "fenbuconazole": ["thuoc diet nam fenbuconazole"],
    "fenclorim": ["thuoc diet co fenclorim"],
    "fenobucarb": ["thuoc diet sau fenobucarb", "fenobucarb"],
    "fertilizer": ["phan bon", "phan hoa hoc"],
    "flonicamid": ["thuoc diet sau flonicamid", "flonicamid"],
    "fluazinam": ["thuoc diet nam fluazinam", "fluazinam"],
    "fludioxonil": ["thuoc diet nam fludioxonil"],
    "fluopicolide": ["thuoc diet nam fluopicolide", "fluopicolide"],
    "flusilazole": ["thuoc diet nam flusilazole", "flusilazole"],
    "forgon-40ec": ["thuoc diet sau forgon 40ec"],
    "forsan-60ec": ["thuoc diet sau forsan 60ec"],
    "fosphite": ["phan fosfit", "phan fosfit truyen dinh duong"],
    "fullkill-50ec": ["thuoc diet sau fullkill 50ec"],
    "fumonisin": ["doc fumonisin", "doc fumonisin tren nong nghiep"],
    "fumonisin-b1": ["doc fumonisin b1"],
    "fumonisins": ["doc fumonisin nhieu loai"],
    "fungi-phite": ["phan bo fungi phite", "phan phan bo fungi phite"],
    "gem": ["thuoc diet sau gem"],
    "gibberellic-acid": ["acid gibberellic", "chat kich thich tang truong gibberellic", "gibberellic", "gibberellic-acid"],
    "glufosinate-amonium": ["thuoc diet co glufosinate amonium", "glu", "glufosinate", "amonium", "glufosinate-amonium"],
    "glufosinate-p": ["thuoc diet co glufosinate p", "glufosinate-p"],
    "graduate-a": ["thuoc diet sau graduate a"],
    "growth-inhibitor": ["chat ngan can phat trien cay", "chat ngan can tang truong"],
    "haloxyfop-p-methyl": ["thuoc diet co haloxyfop p methyl", "haloxyfop", "haloxyfop-p-methyl"],
    "headline": ["thuoc diet sau headline"],
    "heritage": ["thuoc tru sau heritage"],
    "hexaconazole": ["thuoc tru benh hexaconazole", "thuoc tru nam hexaconazole", "hexaconazole"],
    "hexythiazox": ["thuoc tru tru sau hexythiazox", "hexythiazox"],
    "hoagland-solution": ["dung dich hoagland"],
    "hoaglands-solution": ["dung dich hoagland"],
    "hymexazol": ["thuoc tru nam hymexazol", "hymexazol"],
    "ic-top": ["thuoc tru sau ic-top"],
    "imazalil": ["thuoc tru nam imazalil"],
    "imidacloprid": ["thuoc tru sau imidacloprid", "thuoc tru sau imidakloprid", "imidacloprid"],
    "insecticide": ["thuoc diet sau", "thuoc tru sau"],
    "iprodione": ["thuoc tru benh iprodione"],
    "isopropyl-alcohol": ["con isopropanol", "con isopropyl"],
    "isoprothiolane": ["thuoc tru benh isoprothiolane", "isoprothiolane"],
    "javen": ["nuoc javen", "nuoc tay trang javen"],
    "javen-solution": ["nuoc javen", "nuoc tay trang javen"],
    "jingangmycin": ["thuoc khang sinh jingangmycin", "jingangmycin"],
    "k2so4": ["kali sunfat", "phan kali sunfat"],
    "kalibo": ["phan kali kalibo"],
    "kh2po4": ["phan kali dhp", "phan kali dihydrophotphat"],
    "khpo": ["phan kali photphat"],
    "khpo4": ["phan kali photphat"],
    "kings-b-medium": ["medium kings b"],
    "kresoxim-methyl": ["thuoc tru benh kresoxim methyl", "kresoxim", "kresoxim-methyl"],
    "lactic-acid": ["axit lactic"],
    "lactofen": ["thuoc diet co lactofen", "lactofen"],
    "lambda-cyhalothrin": ["thuoc tru sau lambda cyhalothrin", "lambda", "cyhalothrin", "lambda-cyhalothrin"],
    "lan-86": ["thuoc tru benh lan 86"],
    "legion": ["thuoc diet sau legion"],
    "liquid-fertilizer": ["phan bon duong luong", "phan bon loang"],
    "lufenuron": ["thuoc tru sau lufenuron", "lufenuron"],
    "lysol": ["chat tay trung lysol"],
    "mancozeb": ["thuoc tru benh mancozeb", "mancozeb", "manco"],
    "mefenoxam": ["thuoc tru benh mefenoxam"],
    "mesotrione": ["thuoc diet co mesotrione", "mesotrione"],
    "metaflumizone": ["thuoc tru sau metaflumizone", "metaflumizone"],
    "metalaxyl": ["thuoc tru benh metalaxyl", "metalaxyl"],
    "metaldehyde": ["thuoc diet sau kim loai", "thuoc diet sau metaldehyde", "metaldehyde"],
    "metman-bulkl": ["thuoc diet sau metman", "thuoc diet sau metman bulkl"],
    "mgs04": ["phan bon khoang mgs04", "phan bon mgs04"],
    "monosultap": ["monosultap", "thuoc diet sau monosultap"],
    "niclosamide": ["niclosamide", "thuoc diet giun niclosamide"],
    "nitenpyram": ["nitenpyram", "thuoc diet sau nitenpyram"],
    "nivalenol": ["doc to nivalenol", "nivalenol"],
    "nordox": ["nordox", "thuoc diet sau nordox"],
    "npk-15-5-20": ["phan bon npk 15 5 20", "phan bon npk 15-5-20", "npk"],
    "npk-16-16-16": ["phan bon npk 16 16 16", "phan bon npk 16-16-16", "npk"],
    "ochratoxin-a": ["doc to ochratoxin a", "ochratoxin a"],
    "organic-fertilizer": ["phan huu co", "phan huu co trong nong nghiep"],
    "organic-matter": ["chat huu co", "chat huu co trong dat"],
    "oxidizing-agent": ["chat oxy hoa", "tac nhan oxy hoa"],
    "oxine-copper": ["oxine dong", "thuoc diet sau oxine dong"],
    "patulin": ["doc to patulin", "patulin"],
    "pcnb": ["pentachloronitrobenzene", "thuoc diet sau pcnb"],
    "pda": ["mo trung pda", "mo trung pda trong nghien cuu vi khuan"],
    "pda-medium": ["mo trung pda", "mo trung pda trong nghien cuu vi khuan"],
    "peka-29-26": ["phan bon peka 29 26", "phan bon peka 29-26"],
    "penconazole": ["penconazole", "thuoc diet nam penconazole"],
    "penicillin": ["khang sinh penicillin", "penicillin"],
    "pentachloronitrobenzene": ["pcnb", "thuoc diet sau pentachloronitrobenzene"],
    "peptone": ["chat dinh duong peptone", "chat peptone"],
    "permethrin": ["permethrin", "thuoc diet sau permethrin"],
    "pesticides": ["thuoc diet sau", "thuoc tru sau"],
    "petroleum-oil": ["dau mo than", "dau mo than trong nong nghiep", "petroleum-oil"],
    "phenthoate": ["phenthoate", "thuoc diet sau phenthoate"],
    "phosphite": ["phan photphit", "phan photphit trong nong nghiep"],
    "phosphonate": ["phan photphonat", "phan photphonat trong nong nghiep"],
    "phoxim": ["phoxim", "thuoc diet sau phoxim"],
    "phu-gia": ["chat phu gia", "chat phu gia trong nong nghiep"],
    "phytophthora-selective-medium": ["mo trung chon loc phytophthora", "mo trung phytophthora"],
    "pimaricin": ["khang sinh pimaricin", "pimaricin"],
    "pirimiphos-methyl": ["pirimiphos methyl", "thuoc diet sau pirimiphos methyl", "pirimiphos", "pirimiphos-methyl"],
    "polycarbonate": ["nhua polycarbonate", "vat lieu polycarbonate"],
    "potassium-dihydrogen-phosphate": ["phan kali dihydrogen photphat", "phan kali dihydrogen photphat trong nong nghiep"],
    "potassium-nitrate": ["bot kali nitrat", "phan kali nitrat"],
    "potassium-phosphate": ["bot kali photphat", "phan kali photphat"],
    "potato-carrot-agar": ["agar khoai tay ca rot"],
    "potato-dextrose-agar": ["agar khoai tay dextrose"],
    "ppa": ["axit phenylpropionic"],
    "pretilachlor": ["thuoc tru sau pretilachlor", "pretilachlor"],
    "pristine": ["thuoc tru nam pristine"],
    "probiconazole": ["thuoc tru nam probiconazole", "probiconazole"],
    "prochloraz": ["thuoc tru nam prochloraz"],
    "prochloraz-manganese-chloride-complex": ["phuc hop prochloraz mangan clorua"],
    "prochloraz-manganese-complex": ["phuc hop prochloraz mangan", "prochloraz"],
    "prochloraz-manganesse-complex": ["phuc hop prochloraz mangan", "prochloraz", "manganesse"],
    "propamocarb": ["thuoc tru nam propamocarb", "propamocarb"],
    "propamocarb-hcl": ["propamocarb hydrochloride", "propamocarb", "hydrochloride", "propamocarb-hcl"],
    "propiconazole": ["thuoc tru nam propiconazole", "propiconazole"],
    "propoxur": ["thuoc tru sau propoxur", "propoxur"],
    "proteose-peptone": ["proteose pepton"],
    "pymetrozine": ["thuoc tru sau pymetrozine", "pymetrozine"],
    "pyraclostrobin": ["thuoc tru nam pyraclostrobin", "pyraclostrobin"],
    "pyrethroid": ["phan bo pyrethroid", "thuoc tru sau pyrethroid", "pyrethroid", "pyrethroid"],
    "pyridaben": ["thuoc tru sau pyridaben", "pyridaben"],
    "pyriproxyfen": ["thuoc tru sau pyriproxyfen", "pyriproxyfen"],
    "quadris-top": ["thuoc tru nam quadris top"],
    "quicklime": ["voi toi"],
    "quintozene": ["thuoc tru nam quintozene"],
    "quizalofop-p-ethyl": ["quizalofop p ethyl", "quizalofop", "quizalofop-p-ethyl"],
    "r333": ["thuoc tru sau r333"],
    "ridomil": ["thuoc tru nam ridomil"],
    "rose-bengal": ["thuoc nhuom rose bengal"],
    "s-metolachlor": ["thuoc tru sau s metolachlor", "metolachlor", "s-metolachlor"],
    "salegold": ["thuoc tru nam salegold"],
    "salt-solution": ["dung dich muoi"],
    "solvent": ["chat giai phan", "dung moi"],
    "special-additives": ["phu gia dac biet"],
    "specific-fungicides": ["thuoc tru nam dac hieu"],
    "spirodiclofen": ["thuoc tru sau spirodiclofen", "spirodiclofen"],
    "subdue": ["thuoc tru nam subdue"],
    "sucrose": ["duong sucrose"],
    "switch-cyprodinil-fludioxonil": ["thuoc tri nam cyprodinil fludioxonil", "thuoc tri nam switch"],
    "systemic-fungicide": ["thuoc diet nam toan than", "thuoc tri nam toan than"],
    "tebuconazole": ["thuoc tebuconazole", "thuoc tri nam tebuconazole", "tebuconazole"],
    "tembotrione": ["thuoc diet co tembotrione", "tembotrione"],
    "terbuthylazine": ["thuoc diet co terbuthylazine", "terbuthylazine"],
    "terrachlor": ["thuoc diet co terrachlor"],
    "than-duoc-sach-benh": ["than duoc sach benh", "thuoc sach benh"],
    "thiabendazole": ["thuoc thiabendazole", "thuoc tri nam thiabendazole"],
    "thiacloprid": ["thuoc diet sau thiacloprid", "thiacloprid"],
    "thiamethoxam": ["thuoc diet sau thiamethoxam", "thiamethoxam"],
    "thiosultap-sodium": ["thuoc diet sau thiosultap sodium", "thiosultap", "thiosultap-sodium"],
    "thiram": ["thuoc thiram", "thuoc tri nam thiram", "thiram"],
    "thuoc-tru-nam": ["thuoc diet nam", "thuoc tri nam"],
    "thuoc-tru-nhen": ["thuoc diet nhen", "thuoc tru nhen"],
    "tilt": ["thuoc tilt", "thuoc tri nam tilt"],
    "tolfenpyrad": ["thuoc diet sau tolfenpyrad", "thuoc tolfenpyrad", "tolfenpyrad"],
    "topramezone": ["thuoc diet co topramezone", "thuoc topramezone", "topramezone"],
    "toxic-chemical": ["chat doc", "chat doc hai"],
    "triadimefon": ["thuoc tri nam triadimefon", "thuoc triadimefon", "triadimefon"],
    "tricyclazole": ["thuoc tri nam tricyclazole", "thuoc tricyclazole", "tricyclazole"],
    "tricylazole": ["thuoc tri nam tricylazole", "thuoc tricylazole", "tricylazole"],
    "trifloxystrobin": ["thuoc tri nam trifloxystrobin", "thuoc trifloxystrobin"],
    "trinong-50wp": ["thuoc tri nam trinong 50wp", "thuoc trinong 50wp"],
    "trioxystrobin": ["thuoc tri nam trioxystrobin", "thuoc trioxystrobin", "trioxystrobin"],
    "ultra-flourish": ["thuoc phan bon ultra flourish", "thuoc ultra flourish"],
    "wa-0-05": ["thuoc wa 0 05"],
    "zearalenone": ["chat doc zearalenon", "chat doc zearalenone"],
    "zhongshengmycin": ["thuoc khang sinh zhongshengmycin", "thuoc zhongshengmycin", "zhongshengmycin"],
    "xu li hat giong": ["xu li hat giong", "hat giong"],
}

CROP_ALIASES = {
    "apple": ["cay tao", "tao"],
    "avocado": ["qua bo", "cay bo", "trai bo"],
    "bau": ["bau"],
    "banana": ["cay chuoi", "chuoi"],
    "bap": ["bap", "cay bap", "corn", "ngo"],
    "bap-cai": ["bap cai", "cay bap cai"],
    "barley": ["cay lua mi", "lua mi"],
    "bean": ["cay dau", "dau"],
    "kho-qua": ["kho qua", "qua kho qua", "muop dang"],
    "tieu": ["tieu", "tieu den"],
    "bi-dao": ["bi dao", "bi huong"],
    "buoi": ["buoi"],
    "bong-vai": ["bong vai"],
    "ca-chua": ["ca chua", "ca chua bi"],
    "ca-phe": ["ca phe"],
    "rau-cai-thia": ["rau cai thia"],
    "cai-brussel": ["cai brussel"],
    "cai": ["cai"],
    "cai-cu": ["cai cu"],
    "cai-thao": ["cai thao"],
    "can-nuoc": ["can nuoc"],
    "can-tay": ["can tay"],
    "ca-phao": ["ca phao"],
    "cac-loai-dua": ["cac loai dua"],
    "cac-loai-hoa": ["cac loai hoa"],
    "cay-mo": ["cay mo"],
    "cay-rung": ["cay rung"],
    "chanh-day": ["chanh day"],
    "chanh-quyt": ["chanh quyt"],
    "che": ["che"],
    "tra": ["tra"],
    "cu-cai": ["cu cai"],
    "dau-cove": ["dau cove"],
    "dau-den": ["dau den"],
    "dau-tam": ["dau tam"],
    "dat-khong-trong-trot": ["dat khong trong trot"],
    "dat-trong-cay-cong-nghiep": ["dat trong cay cong nghiep", "cay cong nghiep"],
    "dua-le": ["dua le"],
    "le": ["le"],
    "dua-leo": ["dua leo", "dua chuot"],
    "dua-luoi": ["dua luoi"],
    "mac-ca": ["mac ca"],
    "mang-tay": ["mang tay"],
    "mong-toi": ["mong toi"],
    "tan-o": ["tan o"],
    "ca-tim": ["ca tim"],
    "cacao": ["ca cao"],
    "cam": ["cam"],
    "cay-an-qua": ["cam", "quyt", "buoi", "xoai", "cafe", "ca phe"],
    "cam-quyt": ["cam quyt"],
    "cao-su": ["cao su"],
    "carrot": ["ca rot"],
    "san": ["san"],
    "xa-lach": ["xa lach"],
    "thom": ["thom"],
    "tia-to": ["tia to"],
    "cu-toi": ["cu toi"],
    "cay-an-trai": ["cay an trai"],
    "cay-cam": ["cay cam"],
    "cay-con": ["cay con"],
    "cay-trong": ["cay trong"],
    "chanh": ["chanh"],
    "che": ["che"],
    "ca-cao": ["ca cao"],
    "bong-vai": ["bo vai"],
    "cu-dau": ["cu dau", "cu dau nong nghiep"],
    "dau-nanh": ["dau nanh", "hat dau nanh"],
    "dau-phong": ["dau phong", "hat dau phong"],
    "dau-tay": ["dau tay", "hat dau tay"],
    "dau-tuong": ["dau tuong", "hat dau tuong"],
    "dau-xanh": ["dau xanh", "hat dau xanh"],
    "dieu": ["cay dieu", "hat dieu"],
    "dragon-fruit": ["cay thanh long", "thanh long"],
    "du-du": ["cay du du", "du du"],
    "dua": ["cay dua", "dua"],
    # "durian": ["cay sau rieng", "sau rieng"],
    "ca-tim": ["ca tim", "cay ca tim"],
    "hoa": ["cay hoa"],
    "sup-lo": ["sup lo"],
    "gung": ["gung"],
    "hanh": ["hanh"],
    "hanh-la": ["hanh la"],
    "hoa-cuc": ["hoa cuc"],
    "hoa-dao": ["hoa dao"],
    # "hoa-hong": ["hoa hong"],
    "hanh-hoa": ["hanh hoa"],
    "hanh-tay": ["hanh tay"],
    "oi": ["oi"],
    "hoa-hong-dai-multiflora": ["hoa hong dai"],
    "hoa-ly": ["hoa ly"],
    "hoa-mai": ["hoa mai"],
    "mai": ["mai"],
    "jackfruit": ["mit"],
    "khoai-lang": ["khoai lang"],
    "khoai-tay": ["khoai tay"],
    "khoai-mi": ["khoai mi", "khoai san"],
    "khom": ["khom"],
    "lac": ["dau phong", "lac"],
    "chanh-vang": ["chanh vang"],
    "lime": ["chanh tay", "lime"],
    "longan": ["nhan"],
    "nghe": ["nghe"],
    "lua": ["cay lua", "lua"],
    "mai": ["hoa mai"],
    "mang-cau": ["mang cau"],
    "me-vung": ["me vung"],
    "mia": ["mia"],
    "mit": ["mit"],
    "ngo": ["ngo"],
    "nho": ["nho"],
    "ot": ["ot"],
    "pak-choi": ["cai bap"],
    "pea": ["dau ha lan"],
    "peach": ["cay dao", "dao"],
    "peanut": ["dau phong", "hat dau phong"],
    "pear": ["cay le", "le"],
    "pepper": ["cay tieu", "tieu"],
    "dua-hau": ["dua hau", "dua tay"],
    "plum": ["cay man", "man"],
    "potato": ["khoai", "khoai tay"],
    "quyt": ["cay quyt", "quyt"],
    "ra-hoa": ["ra hoa", "thoi ky ra hoa"],
    "quat": ["quat"],
    "tac": ["tac"],
    "rau-cai": ["rau cai", "rau cai la"],
    "rau-muong": ["rau muong"],
    "rau-mau": ["rau mau", "rau mau an"],
    "rau-cu": ["rau cu"],
    "rau-ho-thap-tu": ["rau ho thap tu", "ho thap tu"],
    "rau-den": ["rau den"],
    "rau-ngo": ["rau ngo"],
    "rau-diep": ["rau diep"],
    "rau-trai": ["rau trai"],
    "thai-lai": ["thai lai"],
    "rau-xa-bong": ["rau xa bong"],
    "rau-chan-vit": ["rau chan vit", "chan vit"],
    "resistant-variety": ["giong chong benh", "giong khang benh"],
    "lua-sa": ["lua sa"],
    # "rose": ["cay hoa hong", "hoa hong"],
    "sam-ngoc-linh": ["sam ngoc linh"],
    "san": ["cay san"],
    "su-hao": ["su hao"],
    "sau-rieng": ["cay sau rieng", "sau rieng"],
    "bau-bi": ["bi ngo", "qua bi ngo"],
    "sweet-orange": ["cam ngot"],
    "thanh-long": ["qua thanh long", "thanh long"],
    "thuoc-la": ["cay thuoc la", "thuoc la"],
    "vegetable": ["rau an"],
    "vegetable-crops": ["cay rau"],
    "vegetables": ["rau an"],
    "vuon-cay-an-trai": ["vuon cay an trai"],
    "watermelon": ["dua hau", "qua dua hau"],
    "long-vuc": ["long vuc"],
    "weed": ["co dai", "mac-co", "rau sam", "co cuc", "cho de", "den gai", "co chan vit", "co long vuc", "co man trau"],
    "wheat": ["lua mi"],
    "xoai": ["qua xoai", "xoai"],
}

DISEASE_ALIASES = {
    "nhom-a": 
    ["nhom benh a", "nhom a", "nam nhom a", "nhom nam a", "benh nhom a", "than thu", "dao on", "dom vong", "dom tim", "bi thoi", "chay la", "dom nau", "dom la", "heo ru", "chet cham", "chay day", "thoi re", "lua von", "lem lep hat", "phan trang", "moc xam", "nam long chuot", "ghe la", "ghe trai", "dom den", "thoi than", "thoi hach", "thoi re", "benh thoi canh", "chay canh", "thoi qua", "benh chet canh", "benh scab", "benh ghe", "san vo", "tiem lua", "vang be", "thoi trai", "kho dot", "chet canh", "nut than", "chay nhua", "benh dom nau", "kho", "benh thoi"
    ],
    "nhom-b": 
    ["nhom benh b", "nhom b", "nam nhom b", "nhom nam b", "benh nhom b", "lo co re", "heo cay con", "chay la", "kho van", "nam hong", "heo ru", "moc trang", "co re bi thoi nau", "thoi nau", "thoi nhun", "benh chet rap cay con", "thoi trai", "thoi than", "ri sat", "than hat lua", "benh ri sat dau tuong", "than thu", "dom la lon", "lem lep hat", "benh thoi"
    ],
    "nhom-o": 
    ["nhom benh o", "nhom o", "nam nhom o", "nhom nam o", "benh nhom o", "suong mai", "benh thoi re", "thoi ngon", "thoi mam", "chet nhanh", "thoi trai", "nut than", "xi mu", "vang la", "chet than", "chet canh", "thoi re", "chet cay con", "moc suong", "gia suong mai", "soc trang", "bach tang", "moc xuong", "ri trang", "nam trang", "phong trang", "benh thoi"],
    "ba-trau": ["dom ba trau", "dom nau", "ba trau"],
    "bac-la": ["bac la", ],
    "benh-nam-hoa-vang": ["hoa vang",],
    "chay-bia-la": ["chay bia la", ],
    "chet-day": ["chet day", ],
    "dom-la": ["dom den la", "den la", "dom la"],
    "dom-den-la": ["dom den la", "den la", "dom la"],
    "dom-mat-cua": ["mat cua"],
    "dom-trai": ["dom trai"],
    "dom-van": ["dom van"],
    "dom-vang": ["dom vang"],
    "heo-vang": ["heo vang"],
    "heo-xanh": ["heo xanh"],
    "lem-lep": ["lem lep"],
    "loet": ["loet"],
    "rung-la-mai": ["rung la mai"],
    "rung-la": ["rung la"],
    "ghe": ["ghe"],
    "seo": ["seo"],
    "ghe-khoai": ["ghe khoai"],
    "ghe-cam": ["ghe cam"],
    "ghe-seo": ["ghe seo"],
    "seo-qua": ["seo qua"],
    "seo-trai": ["seo trai"],
    "thoi-co-gie": ["thoi co gie"],
    "thoi-cu": ["thoi cu"],
    "thoi-dau-trai": ["thoi dau trai"],
    "rung-trai": ["rung trai"],
    "thoi-den": ["thoi den"],
    "thoi-goc": ["thoi goc"],
    "thoi-than-xi-mu": ["thoi than xi mu"],
    "u-cuc-re": ["u cuc re"],
    "vang-la-chin-som": ["vang la chin som"],
    "vang-rung-la": ["vang rung la"],
    "xoan-la": ["xoan la"],

}

PEST_ALIASES = {
    "bo-ngau": ["bo ngau", "bo ngau tren cay"],
    "bo-nhay": ["bo nhay", "bo nhay tren cay"],
    "bo-phan": ["bo phan", "bo phan tren cay"],
    "bo-hung": ["bo hung", "bo phan tren cay"],
    "bo-rua": ["bo rua",],
    "bo-canh-to": ["bo canh to",],
    "bu-lach": ["bu lach",],
    "bo-tri-vang": ["bo tri vang",],
    "cay-xau-ho": ["cay xau ho"],
    "chac-lac": ["chac lac"],
    "buom-trang-hai-cham": ["buom trang hai cham",],
    "bo-phan-trang": ["bo phan trang", "bo phan trang tren cay"],
    "bo-tri": ["bo tri", "xu ly tri", "xu ly bo tri", "bo hut", "bo tri tren cay"],
    "bo-xit": ["bo xit", "bo xit tren cay", "bo xit muoi"],
    "bo-ha": ["bo ha", "sung dat", "sung"],
    "borer": ["sau duc than", "sau duc trong"],
    "borers": ["sau duc than", "sau duc trong"],
    "co-la-rong": ["co la rong", "co la rong trong ruong"],
    "co-duoi-chon": ["co duoi chon",],
    "co-dong-tien": ["co dong tien",],
    "co-hoi": ["co hoi",],
    "co-long-hoi": ["co long hoi",],
    "co-la-tre": ["co la tre",],
    "co-cu": ["co cu",],
    "co-lac": ["co lac",],
    "co-muc": ["co muc",],
    "co-gao": ["co gao",],
    "co-ray": ["co ray",],
    "co-tuc": ["co tuc",],
    "co-gau": ["co gau",],
    "co-chac": ["co chac",],
    "co-hoa-ban": ["co hoa ban",],
    "co-hoa-thao": ["co hoa thao",],
    "co-chao": ["co chao", "co chao trong ruong"],
    "co-chi": ["co chi", "co chi trong ruong"],
    "co-dai-la-hep": ["co dai la hep", "co dai la nho", "la hep"],
    "co-dai-la-rong": ["co dai la rong", "co dai la to", "la rong"],
    "co-duoi-phung": ["co duoi phung", "co duoi phung la"],
    "man-trau": ["man trau"],
    "co-man-trau": ["co man trau", "co man trau la", "man-trau"],
    "co-tranh": ["co tranh", "co tranh la"],
    "compacted-soil": ["dat nen cat cung", "dat nen cung"],
    "dao-on": ["benh dao on", "dao on", "dao on co bong"],
    "doi-duc-la": ["benh doi duc la", "doi duc la"],
    "duc-than": ["duc than"],
    "duoi-phung": ["duoi phung"],
    "eggs": ["trung con trung", "trung sau"],
    "sieu-nhan": ["sieu nhan"],
    "tuyen-trung": ["tuyen trung"],
    "glyphosate-resistant-weeds": ["cac loai co khang glyphosate", "co khang", "khang glyphosate"],
    "oc-buou-vang": ["oc buou", "oc buou vang", "oc vang"],
    "nam": ["nam", "nam cay trong"],
    "nhen": ["con nhen", "ruoi nhen", "nhen"],
    "oc-buu-vang": ["oc buu vang", "oc vang","oc"],
    "pests": ["sau benh", "sau hai"],
    "ph": ["do axit kiem", "do ph"],
    "phan-trang": ["benh phan trang", "phan trang"],
    "rat": ["chuot hai cay"],
    "ray": ["ray", "ray tren cay"],
    "ray-bong": ["ray bong", "ray bong tren cay"],
    "lung-trang": ["lung trang",],
    "ray-lung-trang": ["ray lung trang", "ray lung trang tren cay"],
    "ray-mem": ["ray mem", "ray mem tren cay"],
    "ray-nau": ["ray nau", "ray lung trang", "ray nau tren cay"],
    "ray-phan": ["ray phan", "ray phan tren cay"],
    "ray-xanh": ["ray xanh", "ray-chong-canh", "ray phan", "ray mem", "bo phan trang", "bo xit"],
    "rep": ["rep", "rep tren cay"],
    "rep-bong-xo": ["rep bong xo", "rep bong xo tren cay"],
    "rep-mem": ["rep mem", "rep mem tren cay"],
    "rep-muoi": ["rep muoi", "rep muoi tren cay"],
    "rep-sap": ["rep sap", "rep vay", "rep sap tren cay"],
    "rep-vay": ["rep vay", "rep sap", "rep vay tren cay"],
    "ruoi-vang": ["ruoi vang", "ruoi vang lua"],
    "rust-fungus": ["benh nam giang", "nam giang"],
    "sau-benh": ["sau benh", "sau gay benh"],
    "sau-bo": ["sau bo", "sau bo la"],
    "sau-chich-hut": ["sau chich hut", "sau hut mau"],
    "sau-cuon-la": ["sau cuon la", "sau cuon la lua", "cuon la"],
    "sau-duc-hoa": ["sau duc hoa"],
    "sau-duc-ngon": ["sau duc ngon"],
    "sau-keo": ["sau keo"],
    "sau-long": ["sau long"],
    "sau-dat": ["sau dat", "sau dat lua"],
    "sau-nang": ["sau nang"],
    "sau-phao": ["sau phao"],
    "sau-rom": ["sau rom"],
    "sen-nhot": ["sen nhot"],
    "sen-vo-mong": ["sen vo mong"],
    "sung-bo-ha": ["sung bo ha"],
    "sau-xanh-da-lang": ["sau xanh da lang", "da lang", "sau xanh"],
    "sau-duc-be": ["sau duc be", "sau duc than be"],
    "sau-duc-canh": ["sau duc canh", "sau duc canh lua"],
    "sau-duc-qua": ["sau duc qua", "sau duc qua lua"],
    "sau-duc-than": ["sau duc than", "sau duc than lua"],
    "sau-duc-trong": ["sau duc trong", "sau duc trong lua"],
    "sau-hai": ["sau hai", "sau hai lua"],
    "sau-hanh": ["sau hanh", "sau cho hanh", "sau hanh lua"],
    "sau-khoang": ["sau khoang", "sau khoang lua"],
    "sau-mieng-nhai": ["sau mieng nhai", "sau nhai la"],
    "sau-nan": ["sau nan", "sau nan lua"],
    "sau-rieng-fruit-borer": ["sau duc trai qua rieng"],
    "sau-sung-trang": ["sau sung trang", "sau sung trang lua"],
    "sau-to": ["sau to", "sau to tren bap cai", "sau to tren bap su"],
    "sau-tong-hop": ["cac loai sau tong hop", "sau tong hop"],
    "sau-ve-bua": ["sau ve bua", "sau ve bua lua"],
    "sau-xam": ["sau xam", "sau xam la"],
    "sau-xanh": ["sau xanh", "sau xanh la"],
    "slug": ["oc ban", "oc ban trong ruong"],
    "oc-sen": ["oc sen", "oc sen trong ruong"],
    "oc-nhot": ["oc nhot"],
    "sau-bay": ["sau bay"],
    "xi-mu": ["xi mu"],
    "nut-vo": ["nut vo"],
    "nut-than": ["nut than"],
    "sung-khoai": ["sung khoai", "sung khoai tay"],
    "suong-mai": ["benh suong mai", "suong mai"],
    "than-thu": ["sau than", "than thu"],
    "thrips": ["bo tri", "tri tren dua non", "tri tren la"],
    "ve-sau": ["con ve sau", "ve sau"],
    "weeds": ["co dai", "co tranh", "co dai trong lua", "co dai trong ruong"],
    "mac-co": ["mac co"],
    "mat-cua": ["mat cua"],
    "moi": ["moi"],
    "mot-duc-canh": ["mot duc canh", "mot"],
    "muoi-den": ["muoi den"],
    "nam-coc": ["nam coc"],
    "nam-hoa-vang": ["nam hoa vang"],
    "nam-moc-nau": ["nam moc nau"],
    "ray-chong-canh": ["ray chong canh"],
    "tam-bop": ["tam bop"],
}

PRODUCT_ALIASES = {
    "afenzole-top-325sc": ["thuoc diet sau afenzole top", "thuoc tru sau afenzole", "afenzoletop", "2 hoat chat", "hai hoat chat"],
    "amamectin-60": ["thuoc diet sau amamectin", "thuoc tru sau amamectin 60", "amamectin"],
    "anh-hung-sau": ["sau anh hung", "sau hai hai", "anh hung sau"],
    "ankamec-3.6ec": ["thuoc diet sau ankamec", "thuoc tru sau ankamec 3 6ec", "ankamec"],
    "asmilka-top-325sc": ["thuoc diet sau asmilka top", "thuoc tru sau asmilka top 325sc", "asmilka"],
    "atomin-15wp": ["thuoc diet sau atomin", "thuoc tru sau atomin 15wp", "atomin"],
    "benxana-240sc": ["thuoc diet sau benxana", "thuoc tru sau benxana 240sc", "benxana"],
    "bmc-sulfur-80wg": ["thuoc diet nam bmc sulfur", "thuoc tru nam bmc sulfur 80wg", "sulfur"],
    "bo-rung": ["bo rung", "con bo rung"],
    "bong-ra-sai-150": ["thuoc diet sau bong ra sai", "thuoc tru sau bong ra sai 150", "bong ra sai","bongrasai", "bongsai"],
    "dapharnec-3.6ec": ["thuoc diet sau dapharnec", "thuoc tru sau dapharnec 3 6ec", "dapharnec"],
    "downy-650wp": ["thuoc diet nam downy", "thuoc tru nam downy 650wp", "downy"],
    "dum-xpro-650wp": ["thuoc diet nam dum xpro", "thuoc tru nam dum xpro 650wp", "xpro", "dumxpro"],
    "forsan-60ec": ["thuoc diet sau forsan", "thuoc tru sau forsan 60ec", "forsan", "forsan60ec"],
    "g9-thanh-sau": ["thuoc diet sau g9 thanh sau", "thuoc tru sau g9 thanh sau", "thanh sau", "g9"],
    "giao-su-benh-4.0": ["thuoc diet nam giao su benh", "thuoc tru nam giao su benh", "giao su benh"],
    "gone-super-350ec": ["thuoc diet sau gone super", "thuoc tru sau gone super 350ec", "gon super", "gone super", "3 hoat chat", "ba hoat chat"],
    "haihamec": ["thuoc diet sau haihamec", "thuoc tru sau haihamec", "haihamec"],
    "haruko-5sc": ["thuoc diet sau haruko", "thuoc tru sau haruko 5sc", "haruko"],
    "haseidn-gold": ["thuoc diet sau haseidn gold", "thuoc tru sau haseidn gold", "haseidn"],
    "horisan-75": ["thuoc diet nam horisan", "thuoc tru nam horisan 75", "horisan"],
    "kaijo-5.0wg": ["thuoc diet nam kaijo", "thuoc tru nam kaijo 5 0wg", "kaijo"],
    "kajio-1gr-alpha": ["thuoc diet sau kajio 1gr alpha", "thuoc diet sau kajio alpha", "kaijo"],
    "kajio-1gr-gold": ["thuoc diet sau kajio 1gr gold", "thuoc diet sau kajio gold", "kaijo"],
    "khai-hoang-g63": ["thuoc diet sau khai hoang", "thuoc diet sau khai hoang g63", "khai-hoang", "g63"],
    "khai-hoang-p7": ["thuoc diet sau khai hoang", "thuoc diet sau khai hoang p7", "khai-hoang"],
    "khai-hoang-q10": ["thuoc diet sau khai hoang", "thuoc diet sau khai hoang q10", "khai-hoang"],
    "khai-hoang-q7": ["thuoc diet sau khai hoang", "thuoc diet sau khai hoang q7", "khai-hoang"],
    "komulunx-80wg": ["thuoc diet nam komulunx", "thuoc tru nam komulunx 80wg", "komulunx"],
    "koto-240sc": ["thuoc diet sau koto", "thuoc tru sau koto 240sc", "koto"],
    "koto-240sc-gold": ["thuoc diet sau koto gold", "thuoc tru sau koto 240sc gold", "koto gold"],
    "kyodo-25sc": ["thuoc diet sau kyodo", "thuoc tru sau kyodo 25sc", "kyodo"],
    "kyodo-50wp": ["thuoc diet nam kyodo", "thuoc tru nam kyodo 50wp", "kyodo"],
    "lyrhoxini": ["thuoc diet sau lyrhoxini", "thuoc tru sau lyrhoxini", "lyrhoxini"],
    "m8-sing": ["thuoc diet sau m8 sing", "thuoc tru sau m8 sing", "m8 sing"],
    "matscot-500sp": ["thuoc diet sau matscot", "thuoc tru sau matscot 500sp", "matscot"],
    "newfosinate": ["thuoc diet co fosinate moi", "thuoc diet co newfosinate", "newfosinate"],
    "newtongard-75": ["thuoc diet nam newtongard", "thuoc tru nam newtongard 75", "newtongard"],
    "niko-72wp": ["niko 72wp", "thuoc niko 72wp", "thuoc tru sau niko 72wp", "niko"],
    "oosaka-700wp": ["oosaka 700wp", "thuoc oosaka 700wp", "thuoc tru sau oosaka 700wp", "oosaka"],
    "phuong-hoang-lua": ["phuong hoang lua", "thuoc phuong hoang lua"],
    "recxona-350SC": ["recxona 350sc", "thuoc recxona 350sc", "recxona"],
    "scrotlan-80wp-m45-an": ["scrotlan 80wp m45 an", "thuoc scrotlan 80wp m45 an", "scrotlan"],
    "snoil-delta-thuy-si": ["snoil delta thuy si", "thuoc snoil delta thuy si", "snoil", "thuoc oc", "thuoc tru oc", "thuoc diet oc", "thuoc tri oc"],
    "soccarb-80wg": ["soccarb 80wg", "thuoc soccarb 80wg", "soccarb"],
    "teen-super-350ec": ["teen super", "thuoc teen super 350ec"],
    "tosi-30wg": ["thuoc tosi 30wg", "tosi"],
    "trau-den-150sl": ["thuoc trau den 150sl", "trau den"],
    "trau-rung-2-0": ["thuoc trau rung 2 0", "trau rung"],
    "trau-rung-moi": ["thuoc trau rung moi", "trau rung"],
    "trau-vang-280": ["thuoc trau vang 280", "trau vang"],
    "trum-nam-benh": ["thuoc trum nam benh", "trum nam benh", "nam benh"],
    "trum-sau": ["thuoc trum sau", "trum sau"],
    "truong-doi-ky": ["thuoc truong doi ky", "truong doi ky"],
    "vua-imida": ["thuoc vua imida", "vua imida", "imida"],
    "vua-rep-sau-ray": ["thuoc vua rep sau ray", "vua rep"],
    "vua-sau": ["thuoc vua sau", "vua sau"],
    "vua-tri": ["thuoc vua tri", "vua tri"],
    "zeroanvil": ["thuoc zeroanvil", "zeroanvil"],
    "Afenzole": ["thuoc afenzole", "afenzole"],
    "Oosaka": ["thuoc oosaka", "oosaka", "thuoc oc", "thuoc tru oc", "thuoc diet oc", "thuoc tri oc"],
    "Recxona-35WG": ["thuoc recxona 35wg", "recxona"],
    "Sinapyram": ["thuoc sinapyram", "sinapyram"],
    "abamectin-3-6-duc": ["thuoc abamectin 3 6 duc", "thuoc diet sau abamectin", "thuoc abamectin", "abamectin"],
    "abinsec": ["thuoc abinsec", "abinsec"],
    "abinsec-1-8ec": ["thuoc abinsec 1 8ec", "abinsec 1.8ec", "abinsec 1.8", "thuoc abinsec 1.8ec ", "abinsec"],
    "abinsec-emaben-sau-1-8ec": ["thuoc abinsec emaben sau 1 8ec", "abinsec", "emaben"],
    "abinsec-oxatin-1-8ec": ["thuoc abinsec oxatin 1 8ec", "oxatin", "abinsec", "abinsec oxatin", "thuoc abinsec oxatin"],
    "abinsec-sieu-diet-nhen": ["thuoc abinsec diet nhen sieu toc", "sieu diet nhen", "abinsec nhen", "abinsec"],
    "aco-one-40ec": ["thuoc aco one 40ec", "aco one", "aco one 40ec", "aco one"],
    "aco-one400ec": ["thuoc aco one 400ec", "aco one"],
    "afenzole-325sc": ["thuoc afenzole 325sc", "afenzole", "afenzole 325sc", "thuoc afenzole"],
    "amesip-80wp": ["thuoc amesip 80wp", "thuoc amesip", "amesip"],
    "amkamec-3-6-ec": ["thuoc amkamec 3 6 ec", "thuoc amkamec", "thuoc amkamec 3.6ec", "amkamec"],
    "atomin": ["thuoc atomin", "atomin"],
    "atomin-15wp": ["thuoc atomin 15wp", "atomin", "atomin15wp"],
    "azin-rio-45sc": ["thuoc azin rio 45sc", "azinrio", "azin rio"],
    "bac-si-nam-benh-nekko-69wp": ["thuoc bac si nam benh nekko 69wp", "nekko"],
    "bacillus-suv": ["thuoc vi khuan bacillus suv", "vi khuan bacillus suv", "bacillus"],
    "bam-dinh-bmc": ["thuoc bam dinh bmc", "bam dinh"],
    "bao-den": ["bao den"],
    "basuzin-1gb": ["thuoc diet co basuzin", "basuzin"],
    "bilu": ["thuoc diet sau bilu"],
    "binhfos-50ec": ["thuoc diet sau binhfos 50ec", "binhfoc 500ec", "binhfos"],
    "binhfos-50ec-vua-ray-rep": ["thuoc diet ray rep binhfos 50ec", "vua ray rep", "binhfos"],
    "binhfos-50ec-vua-sau": ["thuoc diet sau binhfos 50ec vua", "vua sau", "binhfos"],
    "binhfos-anh-hung-sau": ["thuoc diet sau binhfos anh hung", "anh hung sau", "binhfos"],
    "binhtox-3-8ec": ["thuoc diet sau binhtox 3 8ec", "binhtox", "binh tox"],
    "binhtox-3-8ec-gold": ["thuoc diet sau binhtox 3 8ec vang", "binhtox-gold", "binhtox gold", "binhtox", "binh tox"],
    "binhtox-gold": ["thuoc diet sau binhtox vang", "binhtox-gold", "binhtox gold", "binhtox", "binh tox"],
    "biperin-100ec": ["thuoc diet sau biperin 100ec", "biperin"],
    "bipimai": ["thuoc diet sau bipimai", "bipimai"],
    "bipimai-150ec": ["thuoc diet sau bipimai 150ec", "bipimai", "bipimai 150ec"],
    "bisomin-2sl": ["thuoc diet sau bisomin 2sl", "bisomin"],
    "bisomin-6wp": ["thuoc diet sau bisomin 6wp", "bisomin"],
    "bn-fosthi-10gr": ["thuoc diet sau bn fosthi 10gr", "fosthi", "-fosthi"],
    "bn-meta-18gr": ["thuoc diet sau bn meta 18gr", "meta 18gr"],
    "bpalatox-100ec": ["thuoc diet sau bpalatox 100ec", "bpalatox"],
    "bpsaco-500ec": ["thuoc diet sau bpsaco 500ec", "bpsaco"],
    "bretil-super-300ec": ["thuoc diet sau bretil super 300ec", "bretil"],
    "buti": ["thuoc diet sau buti", "buti"],
    "buti-43sc": ["thuoc diet sau buti 43sc", "buti"],
    "butti-43sc-anh-hung-nhen": ["thuoc diet nhen butti 43sc anh hung", "anh hung nhen", "buti"],
    "byphan-800wp": ["thuoc diet sau byphan 800wp", "byphan"],
    "chessin-600wp": ["thuoc diet sau chessin 600wp", "chessin", "2 hoat chat", "hai hoat chat"],
    "chitin-daphamec-3-6": ["thuoc diet sau chitin daphamec 3 6", "chitin daphamec", "chitin", "chitin 3.6", "daphamec"],
    "chlorfena-240sc": ["thuoc diet sau chlorfena 240sc", "chlorfena"],
    "chowon-550sl": ["thuoc diet sau chowon 550sl than duoc", "chowon", "than duoc"],
    "cocosieu-15-5-wp": ["thuoc diet sau cocosieu 15 5 wp", "cocosieu"],
    "cocosieu-15-5wp": ["thuoc diet sau cocosieu 15 5 wp", "cocosieu"],
    "cymkill-25ec": ["thuoc diet sau cymkill 25ec", "cymkill"],
    "daphamec-5-0ec": ["thuoc diet sau daphamec 5 0ec", "daphamec"],
    "daphamec-5ec": ["thuoc diet sau daphamec 5ec", "daphamec"],
    "diet-nhen-sam-set": ["thuoc diet nhen sam set", "diet nhen sam set", "diet nhen", "sam set"],
    "dolping-40ec": ["thuoc diet sau dolping 40ec", "dolping"],
    "downy": ["thuoc diet benh downy", "downy"],
    "downy-650wp": ["thuoc diet benh downy 650wp", "downy"],
    "dp-avo": ["thuoc dp avo", "dp avo", "dp-avo"],
    "durong-800wp": ["thuoc durong 800wp", "durong", "durong-800wp"],
    "dusan-240ec": ["thuoc dusan 240ec", "dusan", "dusan-240ec"],
    "ecudor-22-4sc": ["thuoc ecudor 22 4sc", "ecudor"],
    "emoil-99ec-spray": ["thuoc em oil 99ec", "Petrolium", "em oil", "em-oil", "oil"],
    "emycin-4wp": ["thuoc emycin 4wp", "emycin", "2 hoat chat", "hai hoat chat"],
    "exami": ["thuoc exami", "exami"],
    "exami-20wg": ["thuoc exami 20wg", "exami"],
    "exami-20wg-ly-tieu-long": ["thuoc exami 20wg ly tieu long", "exami", "ly tieu long"],
    "faptank": ["thuoc faptank", "faptank"],
    "faquatrio-20sl": ["thuoc faquatrio 20sl", "faquatrio"],
    "forcin-50ec": ["thuoc forcin 50ec", "forcin"],
    "forgon-40ec": ["thuoc forgon 40ec", "forgon"],
    "forsan-60ec": ["thuoc forsan 60ec", "forsan"],
    "forsan-horisan": ["thuoc forsan horisan", "forsan horisan", "horisan", "forsan"],
    "fortac-5ec": ["thuoc fortac 5ec", "fortac"],
    "fortazeb-72wp": ["thuoc fortazeb 72wp", "fortazeb"],
    "forthane-80wp": ["thuoc forthane 80wp", "forthane", "forthan"],
    "forwarat-0-005-wax-block": ["thuoc forwarat 0 005 wax block", "forwarat"],
    "forzate-20ec": ["thuoc forzate 20ec", "forzate"],
    "forzate-20ew": ["thuoc forzate 20ew", "forzate"],
    "fuji-boss-30sc": ["thuoc fuji boss 30sc", "fujiboss", "fuji boss", "boss", "fuji", "2 hoat chat", "hai hoat chat"],
    "fujiboss-30sc": ["thuoc fuji boss 30sc", "fujiboss", "fuji boss", "boss", "fuji", "2 hoat chat", "hai hoat chat"],
    "fullkill-50ec": ["thuoc fullkill 50ec", "fullkill", "full kill", "full-kill"],
    "gangter-300ec": ["thuoc gangter 300ec", "gangter"],
    "gardona-250sl": ["thuoc gardona 250sl", "gardona"],
    "giao-su-sau": ["thuoc giao su sau", "giao su sau"],
    "giaosu-co": ["thuoc giao su co", "giao su co"],
    "gibberellic-acid": ["axit gibberelic", "gibberellic"],
    "gly-888": ["thuoc gly 888", "gly", "888", "gly 888", "gly-888"],
    "gone-super": ["thuoc gone super", "gone super"],
    "gone-super-350ec": ["thuoc gone super 350ec", "gone super", "gone super 350ec"],
    "gorop-500ec": ["thuoc gorop 500ec", "gorop"],
    "gussi-bastar-200sl": ["thuoc gussi bastar 200sl", "gussi bastar", "gussi", "bastar"],
    "gussi-bastar-200sl-dac": ["thuoc gussi bastar 200sl dac", "gussi bastar 200sl dac", "gussi bastar", "gussi", "bastar"],
    "haihamec-3-6e": ["thuoc haihamec 3 6e", "haihamec", "haihamec3.6ec", "haihamec3.6"],
    "haihamec-3-6ec": ["thuoc haihamec 3 6ec""haihamec", "haihamec3.6ec", "haihamec3.6"],
    "hariwon-30sl": ["thuoc hariwon 30sl", "hariwon", "hariwon30sl", "hariwon-30sl"],
    "haruko-5sc": ["thuoc bao ve thuc vat haruko 5sc", "thuoc tru sau haruko 5sc", "haruko", "haruko-5sc", "haruko5sc"],
    "hello-fungi-400": ["thuoc bao ve nam hello fungi 400", "thuoc diet nam hello fungi 400", "hello-fungi", "hello fungi", "hello", "fungi"],
    "hexa": ["thuoc bao ve thuc vat hexa", "thuoc hexa", "hexa"],
    "hoanganhvil-50sc": ["thuoc bao ve thuc vat hoang anh vil 50sc", "thuoc hoang anh vil 50sc", "hoanganhvil"],
    "hong-ha-nhi": ["thuoc bao ve thuc vat hong ha nhi", "thuoc hong ha nhi", "hong ha nhi"],
    "hosu-10sc": ["thuoc bao ve thuc vat ho su 10sc", "thuoc ho su 10sc", "hosu10sc", "hosu"],
    "ic-top-28-1sc": ["thuoc bao ve thuc vat ic top 28 1sc", "thuoc ic top 28 1sc", "ictop", "ic-top", "ic top"],
    "ic-top-28-1sc-boocdor": ["thuoc bao ve thuc vat ic top 28 1sc boocdor", "thuoc ic top 28 1sc boocdor", "ictop", "ic-top", "ic top"],
    "igro-240sc": ["thuoc bao ve thuc vat igro 240sc", "thuoc igro 240sc", "igro", "igro240sc"],
    "igro-240sc-ohayo": ["thuoc bao ve thuc vat igro 240sc ohayo", "thuoc igro 240sc ohayo", "igro", "igro240sc"],
    "igro-240sc-xpro": ["thuoc bao ve thuc vat igro 240sc xpro", "thuoc igro 240sc xpro","igro", "igro240sc"],
    "igro-ohayo-240sc": ["thuoc bao ve thuc vat igro ohayo 240sc", "thuoc igro ohayo 240sc", "igro", "igro240sc"],
    "igro-xpro": ["thuoc bao ve thuc vat igro xpro", "thuoc igro xpro", "igro", "igro240sc"],
    "inmanda-100wp": ["thuoc bao ve thuc vat inmanda 100wp", "thuoc inmanda 100wp", "inmanda", "thuoc inmanda", "inmanda"],
    "japasa-50ec": ["thuoc bao ve thuc vat japasa 50ec", "thuoc japasa 50ec", "japasa"],
    "jinhe-barass-0-01sl": ["thuoc bao ve thuc vat jinhe barass 0 01sl", "thuoc jinhe barass 0 01sl", "jinhe-barass", "jinhe barass", "jinhe", "barass"],
    "jinhe-brass-0-01sl": ["thuoc bao ve thuc vat jinhe brass 0 01sl", "thuoc jinhe brass 0 01sl"],
    "kajio-1gr": ["thuoc bao ve thuc vat kajio 1gr", "thuoc kajio 1gr", "kajio 1gr", "kajio"],
    "kajio-1gr-alpha-anh-hung-sung": ["thuoc bao ve thuc vat kajio 1gr alpha anh hung sung", "thuoc kajio 1gr alpha anh hung sung", "kajio alpha", "anh hung sung", "kajio-alpha"],
    "kajio-1gr-gold": ["thuoc bao ve thuc vat kajio 1gr gold", "thuoc kajio 1gr gold", "kajio 1gr gold", "kajio gold", "kajio"],
    "kajio-5ec": ["thuoc bao ve thuc vat kajio 5ec", "thuoc kajio 5ec", "kajio", "kajio-5ec", "kajio5ec"],
    "kajio-5ec-g9-thanh-sau": ["thuoc bao ve thuc vat kajio 5ec g9 thanh sau", "thuoc kajio 5ec g9 thanh sau", "kajio g9", "kajio-g9", "kajio 5ec", "kajio", "kajio-5ec", "kajio thanh sau"],
    "kajio-5wg": ["thuoc bao ve thuc vat kajio 5wg", "thuoc kajio 5wg", "kajio", "kajio-5wg"],
    "kasugamycin": ["thuoc kasugamycin", "thuoc khang sinh kasugamycin", "kasugamycin"],
    "kasuhan-4wp": ["thuoc bao ve thuc vat kasuhan 4wp", "thuoc kasuhan 4wp", "kasuhan 4wp", "kasuhan-4wp", "kasuhan"],
    "kenbast-15sl": ["thuoc bao ve thuc vat kenbast 15sl", "kenbast", "kenbast-15sl"],
    "khai-hoang-g63": ["thuoc bao ve thuc vat khai hoang g63", "thuoc khai hoang g63", "khai hoang g63", "khai hoang"],
    "khai-hoang-q10": ["thuoc bao ve thuc vat khai hoang q10", "thuoc khai hoang q10", "khai hoang q10", "q10"],
    "khai-hoang-q7": ["thuoc bao ve thuc vat khai hoang q7", "thuoc khai hoang q7", "khai hoang q7", "q7"],
    "khongray-54wp": ["thuoc bao ve thuc vat khongray 54wp", "thuoc khongray 54wp", "khongray", "khongray-54wp", "2 hoat chat", "hai hoat chat"],
    "king-cide-japan-460sc": ["thuoc bao ve thuc vat king cide japan 460sc", "thuoc king cide japan 460sc", "king cide japan 460sc", "king cide japan", "king cide", "kingcide", "king-cide", "3 hoat chat", "ba hoat chat"],
    "king-kha-1ec": ["thuoc bao ve thuc vat king kha 1ec", "thuoc king kha 1ec", "king kha 1ec", "king kha", "kinh kha", "kingkha"],
    "koto-240sc": ["thuoc bao ve thuc vat koto 240sc", "thuoc koto 240sc", "koto"],
    "koto-gold-240sc": ["thuoc bao ve thuc vat koto gold 240sc", "thuoc koto gold 240sc", "koto", "koto-gold", "koto gold", "koto-gold"],
    "kyodo": ["thuoc bao ve thuc vat kyodo", "thuoc kyodo", "kyodo"],
    "kyodo-25sc": ["thuoc bao ve thuc vat kyodo 25sc", "thuoc kyodo 25sc", "kyodo"],
    "kyodo-25sc-gold": ["thuoc bao ve thuc vat kyodo 25sc gold", "thuoc kyodo 25sc gold", "kyodo", "kyodo-gold", "kyodo gold"],
    "kyodo-50wp": ["thuoc bao ve thuc vat kyodo 50wp", "thuoc kyodo 50wp", "kyodo"],
    "lac-da": ["thuoc bao ve thuc vat lac da", "thuoc lac da", "lac da"],
    "lama-50ec": ["thuoc bao ve thuc vat lama 50ec", "thuoc lama 50ec", "lama"],
    "lao-ton-108ec": ["lao ton 108 ec", "thuoc lao ton 108 ec", "lao ton"],
    "laoton-108ec": ["lao ton 108 ec", "thuoc lao ton 108 ec", "lao ton"],
    "ledan-4gr": ["ledan 4 gr", "thuoc ledan 4 gr", "ledan", "ledan 4gr", "ledan 4g"],
    "ledan-95sp": ["ledan 95 sp", "thuoc ledan 95 sp", "ledan", "ledan 95sp", "ledan 95"],
    "lekima": ["lekima", "thuoc lekima"],
    "lekima-100ec": ["lekima 100 ec", "thuoc lekima 100 ec", "lekima", "lekima"],
    "lufenuron-5ec": ["lufenuron 5 ec", "thuoc diet sau lufenuron", "thuoc lufenuron 5 ec", "lufenuron"],
    "many-800wp": ["many 800 wp", "thuoc many 800 wp", "many"],
    "maruka-5ec": ["maruka 5 ec", "thuoc maruka 5 ec", "maruka"],
    "matscot": ["matscot", "thuoc matscot"],
    "matscot-50sp": ["matscot", "matscot 50 sp", "thuoc matscot 50 sp"],
    "matscot-50sp-ech-com": ["matscot", "matscot 50 sp ech com", "thuoc matscot 50 sp ech com"],
    "mekongvil-5sc": ["mekongvil", "mekongvil 5 sc", "thuoc mekongvil 5 sc"],
    "mi-stop-350sc": ["mistop", "mi-stop", "mi stop 350 sc", "thuoc mi stop 350 sc"],
    "million-50wg": ["million", "million 50 wg", "thuoc million 50 wg"],
    "miriphos-1gb": ["iriphos", "miriphos 1 gb", "thuoc miriphos 1 gb"],
    "misung-15sc": ["misung", "misung 15 sc", "thuoc misung 15 sc"],
    "mitop-one-390sc": ["mitop-one", "mitop one", "mitop", "thuoc mitop one 390sc", "3 hoat chat", "ba hoat chat"],
    "modusa-960ec": ["modusa", "modusa 960 ec", "thuoc modusa 960 ec"],
    "modusa-960ec-gold": ["modusa", "modusa 960 ec gold", "thuoc modusa 960 ec gold"],
    "nakano-50wp": ["nakano", "nakano 50 wp", "thuoc nakano 50 wp"],
    "napoleon-fortazeb-72wb": ["napoleon", "fortazeb", "napoleon-fortazeb", "napoleon fortazeb 72 wb", "thuoc napoleon fortazeb 72 wb"],
    "naticur": ["naticur", "thuoc naticur", "2 hoat chat", "hai hoat chat"],
    "nekko-69wp": ["nekko 69 wp", "thuoc nekko 69 wp", "nekko"],
    "newfosinate-150sl": ["newfosinate 150 sl", "thuoc newfosinate 150 sl", "newfosinate"],
    "nhen-kim-cuong": ["kim cuong", "thuoc diet nhen kim cuong"],
    "nhen-do": ["nhen do"],
    "nhen-gie": ["nhen gie"],
    "nhen-long-nhung": ["nhen long nhung"],
    "nhen-trang": ["nhen trang"],
    "nhen-vang": ["nhen vang"],
    "niko": ["niko", "thuoc niko"],
    "niko-72wp": ["niko 72 wp", "thuoc niko 72 wp", "niko"],
    "nofara": ["nofara", "thuoc nofara"],
    "nofara-350sc": ["nofara 350 sc", "thuoc nofara 350 sc", "nofara"],
    "nofara-35wg": ["nofara 35 wg", "thuoc nofara 35 wg", "nofara"],
    "oc-15gr": ["oc 15 gr", "thuoc oc 15 gr", "oc 15gr", "thuoc oc", "thuoc tru oc", "thuoc diet oc", "thuoc tri oc"],
    "oc-ly-tieu-long-18gr": ["oc ly tieu long 18 gr", "thuoc oc ly tieu long 18 gr", "oc ly tieu long", "ly tieu long 18gr", "thuoc oc", "thuoc tru oc", "thuoc diet oc", "thuoc tri oc"],
    "ohayo-100sc": ["ohayo 100 sc", "thuoc ohayo 100 sc", "ohayo 100sc", "ohayo"],
    "ohayo-240sc": ["ohayo 240 sc", "thuoc ohayo 240 sc", "ohayo 240sc", "ohayo"],
    "onehope": ["onehope", "thuoc onehope"],
    "onehope-480sl": ["onehope 480sl" ,"onehope 480 sl", "thuoc onehope 480 sl", "onehope"],
    "oosaka-700wp": ["oosaka", "oosaka 700 wp", "thuoc oosaka 700 wp"],
    "oscare-100wp": ["oosaka", "oscare 100 wp", "thuoc oscare 100 wp"],
    "oscare-600wg": ["oosaka", "oscare 600 wg", "thuoc oscare 600 wg"],
    "oxatin": ["oxatin", "thuoc oxatin"],
    "oxine-copper": ["dong oxin", "thuoc dong oxin", "oxine", "copper"],
    "panda-4gr": ["phan panda", "phan panda 4gr", "panda 4gr", "panda-4gr", "panda"],
    "parato-200sl": ["thuoc parato", "thuoc parato 200sl", "parato"],
    "parato-than-lua": ["thuoc parato chong than lua", "thuoc parato than lua", "parato"],
    "paskin-250": ["thuoc paskin", "thuoc paskin 250", "paskin"],
    "phonix-dragon-20ec": ["thuoc phonix dragon", "thuoc phonix dragon 20ec","phonix dragon", "phonix", "dragon"],
    "phuong-hoang-lua": ["thuoc phuong hoang", "thuoc phuong hoang lua", "phuong hoang lua", "phuong hoang"],
    "pilot-15ab": ["thuoc pilot", "thuoc pilot 15ab", "pilot", "thuoc oc", "thuoc tru oc", "thuoc diet oc", "thuoc tri oc"],
    "pim-pim-75wp": ["pim pim", "thuoc pim pim 75wp", "pimpim", "pim-pim"],
    "probicol-200wp": ["thuoc probicol", "thuoc probicol 200wp", "probicol", "propicol", "2 hoat chat", "hai hoat chat"],
    "prochloraz-manganese-50-wp": ["thuoc prochloraz mangan 50wp", "thuoc prochloraz manganese", "prochloraz"],
    "pyrolax-250ec": ["thuoc pyrolax", "thuoc pyrolax 250ec", "pyrolax"],
    "ram-te-thien": ["thuoc ram te", "thuoc ram te thien", "te thien"],
    "raynanusa-400wp": ["thuoc raynanusa", "thuoc raynanusa 400wp", "raynanusa"],
    "riceup-300ec": ["thuoc riceup", "thuoc riceup 300ec", "riceup"],
    "sam-san-2-5sc": ["thuoc sam san", "thuoc sam san 2 5sc", "sam san", "sam-san"],
    "sanbang-30sc": ["thuoc sanbang", "thuoc sanbang 30sc", "sanbang", "san bang", "sangbang", "sang bang"],
    "santoso-100sc": ["thuoc santoso", "thuoc santoso 100sc", "santoso"],
    "scorcarb-80wg": ["thuoc scorcarb", "thuoc scorcarb 80wg", "scorcarb"],
    "scortlan": ["thuoc scortlan", "scortlan"],
    "scortlan-80wp": ["thuoc scortlan 80wp", "scortlan"],
    "setis-34sc": ["thuoc setis", "thuoc setis 34sc", "setis"],
    "setis-giao-su-nhen-34sc": ["thuoc setis giao su nhen", "thuoc setis giao su nhen 34sc", "giao su nhen", "setis"],
    "sha-chong-jing": ["thuoc sha chong jing", "sha chong jing", "sha chong"],
    "sha-chong-jing-95wp": ["thuoc sha chong jing 95wp", "sha chong jing", "sha chong"],
    "shina-18sl": ["thuoc shina", "thuoc shina 18sl", "shina"],
    "shinawa-400ec": ["thuoc shinawa", "thuoc shinawa 400ec", "shinawa"],
    "shonam-500sc": ["thuoc shonam", "thuoc shonam 500sc", "shonam"],
    "showbiz": ["thuoc showbiz", "showbiz"],
    "showbiz-16sc": ["thuoc showbiz 16sc", "showbiz"],
    "sieu-bam-dinh": ["thuoc bam dinh", "thuoc sieu bam dinh", "sieu bam dinh", "bam dinh"],
    "sieu-diet-chuot": ["thuoc diet chuot", "thuoc sieu diet chuot", "diet chuot"],
    "sieu-diet-mam": ["thuoc diet mam", "thuoc sieu diet mam", "diet mam"],
    "sieu-diet-nhen": ["thuoc diet nhen", "thuoc sieu diet nhen", "diet nhen"],
    "sieu-diet-sau": ["thuoc diet sau", "thuoc sieu diet sau", "diet-sau"],
    "sinapy-ram": ["thuoc sinapy", "thuoc sinapy ram", "sinapy", "sinapyram", "ram"],
    "sinapyram-80wg": ["thuoc sinapyram 80wg", "sinapy", "sinapyram", "ram"],
    "somethrin-10ec": ["thuoc somethrin", "thuoc somethrin 10ec", "somethrin"],
    "su-tu-do": ["thuoc su tu do", "su tu do"],
    "suparep-22-4sc": ["thuoc suparep 22 4sc", "thuoc suparep", "suparep"],
    "suparep-400wp": ["thuoc suparep 400wp", "suparep"],
    "supermario-70sc": ["thuoc supermario 70sc", "supermario", "2 hoat chat", "hai hoat chat"],
    "suria-10gr": ["thuoc suria 10gr", "suria"],
    "suron-800wp": ["thuoc suron 800wp", "suron"],
    "takiwa-22sc": ["thuoc takiwa 22sc", "takiwa"],
    "tamatras": ["thuoc tamatras", "tamatras", "spirotetramat tolfenpyrad", "spirotetramat", "tolfenpyrad"],
    "tamiko-50ec": ["thuoc tamiko 50ec", "tamiko"],
    "tatsu-25wp": ["thuoc tatsu 25wp", "tatsu"],
    "tembo-8od-vua-co-ngo": ["thuoc tembo 8od vua co ngo", "tembo", "vua co ngo", "8od"],
    "thalonil-75wp": ["thuoc thalonil 75wp", "thalonil"],
    "thuoc-bilu": ["thuoc bilu"],
    "thuoc-chuot-forwarat-0-005-wax-block": ["thuoc chuot forwarat 0 005 wax block"],
    "tomi": ["thuoc tomi", "tomi"],
    "tomi-5ec": ["thuoc tomi 5ec", "tomi"],
    "topmesi-40sc": ["thuoc topmesi 40sc", "topmesi"],
    "topxapy-30sc": ["thuoc topxapy 30sc", "topxapy"],
    "topxim-pro-30sc": ["thuoc topxim pro 30sc", "topxim", "topximpro"],
    "toshiro-10ec": ["thuoc toshiro 10ec", "toshiro"],
    "tosi": ["thuoc tosi", "tosi"],
    "tosi-30wg": ["thuoc tosi 30wg", "tosi"],
    "trau-den-150": ["thuoc trau den 150", "trau den"],
    "trau-rung-2.0": ["thuoc trau rung 2 0", "trau rung", "trau rung2.0"],
    "trau-rung-moi": ["thuoc trau rung moi", "trau rung"],
    "trau-vang": ["thuoc trau vang", "trau vang"],
    "trinong-50wp": ["thuoc trinong 50wp", "trinong"],
    "trum-chich-hut-tri": ["thuoc trum chich hut tri", "trum chich hut", "chich hut"],
    "truong-vo-ky": ["thuoc truong vo ky", "truong vo ky", "yosky"],
    "uchong-40ec": ["thuoc uchong 40ec", "uchong"],
    "vamco-480sl": ["thuoc vamco 480sl", "vam co", "vamco", "dktazole", "dktazone"],
    "vam-co-dktazole-480sl": ["thuoc vam co dktazole 480sl", "vam co", "dktazole", "vamco"],
    "vet-xanh": ["thuoc vet xanh", "vetxanh", "vet-xanh", "vet xanh"],
    "voi-rung": ["thuoc voi rung", "voi rung"],
    "voi-thai-3-6ec-gold": ["thuoc voi thai 3 6ec gold", "voi thai"],
    "vua-co-ngo": ["thuoc vua co ngo", "vua co ngo"],
    "vua-lua-chay-khai-hoang-malaysia": ["thuoc vua lua chay khai hoang malaysia", "vua lua chay", "khai hoang malaysia", "malaysia"],
    "vua-mida-phuong-hoang-lua": ["thuoc vua mida phuong hoang lua", "vua mida", "phuong hoang lua", "mida"],
    "wusso": ["thuoc wusso", "wusso"],
    "xie-xie-200wp": ["thuoc xie xie 200wp", "xie xie", "xie-xie", "xiexie", "xie"],
    "xiexie-200wp-anh-hung-khuan": ["thuoc diet khuan xiexie 200wp anh hung", "anh hung khuan", "xie xie", "xie-xie", "xiexie", "xie"],
    "yosky": ["thuoc yosky", "yosky, 10sl"],
    "yosky-10sl-khai-hoang-p7": ["thuoc yosky 10sl khai hoang p7", "yosky"],
    "zigen": ["thuoc zigen", "zigen"],
    "zigen-15sc": ["thuoc zigen 15sc", "zigen"],
    "zigen-super": ["thuoc zigen super", "zigen"],
    "zigen-super-15sc": ["thuoc zigen super 15sc", "zigen"],
    "zigen-xpro": ["thuoc zigen xpro", "zigen"],
}

BRAND_ALIASES = {
    "bmc": ["san pham bmc", "cong ty bmc", "cua bmc"],
    "phuc-thinh": ["san pham phuc thinh", "cong ty phuc thinh", "cua phuc thinh"],
    "agrishop": ["san pham agrishop", "cong ty agrishop", "cua agrishop"],
    "delta": ["san pham delta", "cong ty delta", "cua delta"],
}

MECHANISMS_ALIASES = {
    "tiep-xuc-luu-dan-manh": ["tiep xuc va luu dan manh", " tiep xuc luu dan manh", "tiep xuc, luu dan manh", "tiep xuc va luu dan nao manh", " tiep xuc luu dan nao manh", "tiep xuc, luu dan manh", "luu dan manh, tiep xuc", "tiep xuc + luu dan manh", "luu dan manh + tiep xuc", "manh"],
    "tiep-xuc-luu-dan": ["tiep xuc va luu dan", "tiep xuc luu dan", "tiep xuc va luu dan nao", "tiep xuc luu dan nao", "tiep xuc, luu dan", "luu dan, tiep xuc", "tiep xuc + luu dan", "luu dan + tiep xuc nao"],
    "luu-dan-manh": ["luu dan manh", "luu dan nao manh"],
    "tiep-xuc-manh": ["tiep xuc manh", "tiep xuc nao manh"],
    "xong-hoi-manh": ["xong hoi manh", "xong hoi nao manh"],
    "luu-dan": ["luu dan", "lu dan", "luu dan nao", "lu dan nao"],
    "tiep-xuc": ["tiep xuc", "tiep suc", "tiep xuc nao", "tiep suc nao"],
    "xong-hoi": ["xong hoi"],
    "co-chon-loc": ["co chon loc", "bao trum", "trum", "phu", "lua"],
    "khong-chon-loc": ["khong chon loc", "k chon loc"],
}

FORMULA_ALIASES = {
    "cong-thuc-ray-nau": [
        "cong thuc tru ray nau",
        "cong thuc diet ray nau",
        "cong thuc phong tru ray nau",
        "cong thuc tri ray nau",
        "cong thuc thuoc tru ray nau",
        "cong thuc thuoc diet ray nau",
        "cong thuc thuoc phong tru ray nau",
        "cong thuc thuoc tri ray nau",
    ],
    "cong-thuc-ray-lung-trang": [
        "cong thuc tru ray lung trang",
        "cong thuc diet ray lung trang",
        "cong thuc phong tru ray lung trang",
        "cong thuc tri ray lung trang",
        "cong thuc thuoc tru ray lung trang",
        "cong thuc thuoc diet ray lung trang",
        "cong thuc thuoc phong tru ray lung trang",
        "cong thuc thuoc tri ray lung trang",
    ],
    "cong-thuc-ray-xanh": [
        "cong thuc tru ray xanh",
        "cong thuc diet ray xanh",
        "cong thuc phong tru ray xanh",
        "cong thuc tri ray xanh",
        "cong thuc thuoc tru ray xanh",
        "cong thuc thuoc diet ray xanh",
        "cong thuc thuoc phong tru ray xanh",
        "cong thuc thuoc tri ray xanh",
    ],
    "cong-thuc-ray-chong-canh": [
        "cong thuc tru ray chong canh",
        "cong thuc diet ray chong canh",
        "cong thuc phong tru ray chong canh",
        "cong thuc tri ray chong canh",
        "cong thuc thuoc tru ray chong canh",
        "cong thuc thuoc diet ray chong canh",
        "cong thuc thuoc phong tru ray chong canh",
        "cong thuc thuoc tri ray chong canh",
    ],
    "cong-thuc-ray-phan": [
        "cong thuc tru ray phan",
        "cong thuc diet ray phan",
        "cong thuc phong tru ray phan",
        "cong thuc tri ray phan",
        "cong thuc thuoc tru ray phan",
        "cong thuc thuoc diet ray phan",
        "cong thuc thuoc phong tru ray phan",
        "cong thuc thuoc tri ray phan",
    ],
    "cong-thuc-ray-mem": [
        "cong thuc tru ray mem",
        "cong thuc diet ray mem",
        "cong thuc phong tru ray mem",
        "cong thuc tri ray mem",
        "cong thuc thuoc tru ray mem",
        "cong thuc thuoc diet ray mem",
        "cong thuc thuoc phong tru ray mem",
        "cong thuc thuoc tri ray mem",
    ],
    "cong-thuc-bo-phan-trang": [
        "cong thuc tru bo phan trang",
        "cong thuc diet bo phan trang",
        "cong thuc phong tru bo phan trang",
        "cong thuc tri bo phan trang",
        "cong thuc thuoc tru bo phan trang",
        "cong thuc thuoc diet bo phan trang",
        "cong thuc thuoc phong tru bo phan trang",
        "cong thuc thuoc tri bo phan trang",
    ],
    "cong-thuc-bo-xit": [
        "cong thuc tru bo xit",
        "cong thuc diet bo xit",
        "cong thuc phong tru bo xit",
        "cong thuc tri bo xit",
        "cong thuc thuoc tru bo xit",
        "cong thuc thuoc diet bo xit",
        "cong thuc thuoc phong tru bo xit",
        "cong thuc thuoc tri bo xit",
    ],
    "cong-thuc-rep-sap": [
        "cong thuc tru rep sap",
        "cong thuc diet rep sap",
        "cong thuc phong tru rep sap",
        "cong thuc tri rep sap",
        "cong thuc thuoc tru rep sap",
        "cong thuc thuoc diet rep sap",
        "cong thuc thuoc phong tru rep sap",
        "cong thuc thuoc tri rep sap",
    ],
    "cong-thuc-rep-vay": [
        "cong thuc tru rep vay",
        "cong thuc diet rep vay",
        "cong thuc phong tru rep vay",
        "cong thuc tri rep vay",
        "cong thuc thuoc tru rep vay",
        "cong thuc thuoc diet rep vay",
        "cong thuc thuoc phong tru rep vay",
        "cong thuc thuoc tri rep vay",
    ],
    "cong-thuc-bo-tri": [ 
        "cong thuc tru bo tri", 
        "cong thuc diet bo tri", 
        "cong thuc phong tru bo tri", 
        "cong thuc tri bo tri", 
        "cong thuc thuoc tru bo tri", 
        "cong thuc thuoc diet bo tri", 
        "cong thuc thuoc phong tru bo tri", 
        "cong thuc thuoc tri bo tri", 
    ],
    "cong-thuc-sau-to": [
        "cong thuc tru sau to",
        "cong thuc diet sau to",
        "cong thuc phong tru sau to",
        "cong thuc tri sau to",
        "cong thuc thuoc tru sau to",
        "cong thuc thuoc diet sau to",
        "cong thuc thuoc phong tru sau to",
        "cong thuc thuoc tri sau to",
    ],
    "cong-thuc-sau-hanh": [
        "cong thuc tru sau hanh",
        "cong thuc diet sau hanh",
        "cong thuc phong tru sau hanh",
        "cong thuc tri sau hanh",
        "cong thuc thuoc tru sau hanh",
        "cong thuc thuoc diet sau hanh",
        "cong thuc thuoc phong tru sau hanh",
        "cong thuc thuoc tri sau hanh",
    ],
    "cong-thuc-bo-nhay": [
        "cong thuc tru bo nhay",
        "cong thuc diet bo nhay",
        "cong thuc phong tru bo nhay",
        "cong thuc tri bo nhay",
        "cong thuc thuoc tru bo nhay",
        "cong thuc thuoc diet bo nhay",
        "cong thuc thuoc phong tru bo nhay",
        "cong thuc thuoc tri bo nhay",
    ],
    "cong-thuc-sau-cuon-la": [
        "cong thuc tru sau cuon la",
        "cong thuc diet sau cuon la",
        "cong thuc phong tru sau cuon la",
        "cong thuc tri sau cuon la",
        "cong thuc thuoc tru sau cuon la",
        "cong thuc thuoc diet sau cuon la",
        "cong thuc thuoc phong tru sau cuon la",
        "cong thuc thuoc tri sau cuon la",
    ],
    "cong-thuc-sau-duc-than": [
        "cong thuc tru sau duc than",
        "cong thuc diet sau duc than",
        "cong thuc phong tru sau duc than",
        "cong thuc tri sau duc than",
        "cong thuc thuoc tru sau duc than",
        "cong thuc thuoc diet sau duc than",
        "cong thuc thuoc phong tru sau duc than",
        "cong thuc thuoc tri sau duc than",
    ],
    "cong-thuc-sau-duc-than": [
        "cong thuc tru sau duc than",
        "cong thuc diet sau duc than",
        "cong thuc phong tru sau duc than",
        "cong thuc tri sau duc than",
        "cong thuc thuoc tru sau duc than",
        "cong thuc thuoc diet sau duc than",
        "cong thuc thuoc phong tru sau duc than",
        "cong thuc thuoc tri sau duc than",
    ],
    "cong-thuc-sau-duc-qua": [
        "cong thuc tru sau duc qua",
        "cong thuc diet sau duc qua",
        "cong thuc phong tru sau duc qua",
        "cong thuc tri sau duc qua",
        "cong thuc thuoc tru sau duc qua",
        "cong thuc thuoc diet sau duc qua",
        "cong thuc thuoc phong tru sau duc qua",
        "cong thuc thuoc tri sau duc qua",
    ],
    "cong-thuc-sau-ve-bua": [
        "cong thuc tru sau ve bua",
        "cong thuc diet sau ve bua",
        "cong thuc phong tru sau ve bua",
        "cong thuc tri sau ve bua",
        "cong thuc thuoc tru sau ve bua",
        "cong thuc thuoc diet sau ve bua",
        "cong thuc thuoc phong tru sau ve bua",
        "cong thuc thuoc tri sau ve bua",
    ],
    "cong-thuc-sung": [
        "cong thuc tru sung",
        "cong thuc diet sung",
        "cong thuc phong tru sung",
        "cong thuc tri sung",
        "cong thuc thuoc tru sung",
        "cong thuc thuoc diet sung",
        "cong thuc thuoc phong tru sung",
        "cong thuc thuoc tri sung",
    ],
    "cong-thuc-bo-ha": [
        "cong thuc tru bo ha",
        "cong thuc diet bo ha",
        "cong thuc phong tru bo ha",
        "cong thuc tri bo ha",
        "cong thuc thuoc tru bo ha",
        "cong thuc thuoc diet bo ha",
        "cong thuc thuoc phong tru bo ha",
        "cong thuc thuoc tri bo ha",
    ],
    "cong-thuc-nhen": [
        "cong thuc tru nhen",
        "cong thuc diet nhen",
        "cong thuc phong tru nhen",
        "cong thuc tri nhen",
        "cong thuc thuoc tru nhen",
        "cong thuc thuoc diet nhen",
        "cong thuc thuoc phong tru nhen",
        "cong thuc thuoc tri nhen",
    ],
    "cong-thuc-nhen-khang-cao": [
        "cong thuc tru nhen khang cao",
        "cong thuc diet nhen khang cao",
        "cong thuc phong tru nhen khang cao",
        "cong thuc tri nhen khang cao",
        "cong thuc thuoc tru nhen khang cao",
        "cong thuc thuoc diet nhen khang cao",
        "cong thuc thuoc phong tru nhen khang cao",
        "cong thuc thuoc tri nhen khang cao",
    ],
    "cong-thuc-oc-buu-vang": [
        "cong thuc tru oc buu vang",
        "cong thuc diet oc buu vang",
        "cong thuc phong tru oc buu vang",
        "cong thuc tri oc buu vang",
        "cong thuc thuoc tru oc buu vang",
        "cong thuc thuoc diet oc buu vang",
        "cong thuc thuoc phong tru oc buu vang",
        "cong thuc thuoc tri oc buu vang",
    ],
    "cong-thuc-oc-sen": [
        "cong thuc tru oc sen",
        "cong thuc diet oc sen",
        "cong thuc phong tru oc sen",
        "cong thuc tri oc sen",
        "cong thuc thuoc tru oc sen",
        "cong thuc thuoc diet oc sen",
        "cong thuc thuoc phong tru oc sen",
        "cong thuc thuoc tri oc sen",
    ],
    "cong-thuc-oc-ma": [
        "cong thuc tru oc ma",
        "cong thuc diet oc ma",
        "cong thuc phong tru oc ma",
        "cong thuc tri oc ma",
        "cong thuc thuoc tru oc ma",
        "cong thuc thuoc diet oc ma",
        "cong thuc thuoc phong tru oc ma",
        "cong thuc thuoc tri oc ma",
    ],
    "cong-thuc-oc-nhot": [
        "cong thuc tru oc nhot",
        "cong thuc diet oc nhot",
        "cong thuc phong tru oc nhot",
        "cong thuc tri oc nhot",
        "cong thuc thuoc tru oc nhot",
        "cong thuc thuoc diet oc nhot",
        "cong thuc thuoc phong tru oc nhot",
        "cong thuc thuoc tri oc nhot",
    ],
}


INTENT_ALIAS_GROUPS = {
    "formula": FORMULA_ALIASES,
    "mechanisms": MECHANISMS_ALIASES,
    "brand": BRAND_ALIASES,
    "product": PRODUCT_ALIASES,
    "chemical": CHEMICAL_ALIASES
}

# ===========================
# 4) MATCHING ENGINE
# ===========================

def match_formula_alias(query: str, alias_dict: Dict[str, List[str]]) -> Optional[str]:
    for key, patterns in alias_dict.items():
        for p in patterns:
            if p in query:
                return key
    return None

def match_aliases(text, aliases):
    found = set()
    text = normalize(text)

    for key, variants in aliases.items():
        for v in variants:
            alias = normalize(v)
            pattern = rf"(?:^|\s){re.escape(alias)}(?:\s|$)"

            if re.search(pattern, text):
                found.add(key)
                break

    return found


# ===========================
# 5) INFER CHEMICAL FROM KB
# ===========================infer_chemicals_from_kb

def infer_chemicals_from_kb(crops: Set[str], diseases: Set[str], pests: Set[str]) -> Set[str]:
    result = set()

    # Normalize input entities để match với KB (KB đang dùng normalize(x) -> space form)
    crops_n = {normalize(x) for x in crops}
    diseases_n = {normalize(x) for x in diseases}
    pests_n = {normalize(x) for x in pests}

    for chem, data in CHEMICAL_KB.items():
        kb_crops = set(data.get("crops", []))
        kb_diseases = set(data.get("diseases", []))
        kb_pests = set(data.get("pests", []))

        if pests_n and pests_n.intersection(kb_pests):
            result.add(chem)

        if crops_n and diseases_n and crops_n.intersection(kb_crops) and diseases_n.intersection(kb_diseases):
            result.add(chem)

        if crops_n and crops_n.intersection(kb_crops):
            result.add(chem)

        if diseases_n and diseases_n.intersection(kb_diseases):
            result.add(chem)

    return result


# ===========================
# 6) MAIN EXTRACTION
# ===========================

def extract_tags(query: str) -> Dict:

    norm_query = normalize(query)

    crops = match_aliases(norm_query, CROP_ALIASES)
    diseases = match_aliases(norm_query, DISEASE_ALIASES)
    pests = match_aliases(norm_query, PEST_ALIASES)

    direct_chems = match_aliases(norm_query, CHEMICAL_ALIASES)

    kb_chems = infer_chemicals_from_kb(crops, diseases, pests)

    all_chems = direct_chems.union(kb_chems)

    must_tags = set()
    any_tags = set()

    # ENTITY chỉ vào ANY
    for c in crops:
        any_tags.add(f"crop:{c}")

    for d in diseases:
        any_tags.add(f"disease:{d}")

    for p in pests:
        any_tags.add(f"pest:{p}")

    for chem in all_chems:
        any_tags.add(f"chemical:{chem}")

    found_entities = {
        "crops": list(crops),
        "diseases": list(diseases),
        "pests": list(pests),
        "chemicals": list(all_chems)
    }

    return {
        "must": list(must_tags),
        "any": list(any_tags),
        "found": found_entities
    }



# ===========================
# 7) ANSWER MODE DETECTION
# ===========================

PRODUCT_SIGNAL = re.compile(r"(?:^|\s)(thuoc|san pham)(?:\s|$)", re.I)


def detect_answer_mode(query: str, tags: Dict) -> str:

    q = normalize(query)

    found = tags.get("found", {})

    if found.get("pests"):
        return "pest"

    if found.get("diseases"):
        return "disease"

    if found.get("crops"):
        return "crop"

    if found.get("chemicals"):
        return "chemical"

    if PRODUCT_SIGNAL.search(q):
        return "product"

    return "general"


# ===========================
# 8) PUBLIC INTERFACE
# ===========================

def tag_filter_pipeline(query: str) -> Dict:

    norm = normalize(query)

    # 1) Trích xuất entity cơ bản
    tags = extract_tags(query)

    detected_intents = []

    # 2) CHỈ các alias đặc biệt mới được vào MUST
    INTENT_ALIAS_GROUPS = {
        "formula": FORMULA_ALIASES,
        "mechanisms": MECHANISMS_ALIASES,
        "brand": BRAND_ALIASES,
        "product": PRODUCT_ALIASES,
        "chemical": CHEMICAL_ALIASES
    }

    for tag_type, alias_map in INTENT_ALIAS_GROUPS.items():

        matched = match_formula_alias(norm, alias_map)

        if matched:
            tags["must"].append(f"{tag_type}:{matched}")
            detected_intents.append(tag_type)

    # 3) Chuyển toàn bộ các tag crop / pest / disease sang ANY (không MUST)
    # extract_tags hiện đang đưa crop/pest/disease vào MUST → cần chuyển lại

    refined_must = []
    refined_any = set(tags["any"])

    for t in tags["must"]:
        if t.startswith("crop:") or t.startswith("pest:") or t.startswith("disease:"):
            refined_any.add(t)
        else:
            refined_must.append(t)

    tags["must"] = refined_must
    tags["any"] = list(refined_any)

    # 4) Quyết định answer_mode

    if detected_intents:
        mode = detected_intents[0]
    else:
        mode = detect_answer_mode(query, tags)

    return {
        "query": query,
        "must": tags["must"],
        "any": tags["any"],
        "found": tags["found"],
        "answer_mode": mode
    }

# ===========================
# TEST
# ===========================

if __name__ == "__main__":
    q = "Thuốc trị mọt đục cành."

    print(json.dumps(tag_filter_pipeline(q), indent=2, ensure_ascii=False))
