import re
import unicodedata
from rag.debug_log import debug_log
from typing import Dict, List, Tuple, Set, Any, Union, Optional

# ======================
# 1) NORMALIZE
# ======================

_space_re = re.compile(r"\s+")

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("đ", "d") 
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = _space_re.sub(" ", s)
    return s


# ======================
# 2) YOUR ALIASES (bổ sung dần)
# ======================
CHEMICAL_ALIASES = {
    "24-epi-brassinolide": ["brassinolid 24-epi", "hooc mon brassinolide 24-epi", "brassinolide"],
    "24-epibrassinolide": ["brassinolid 24-epi", "hooc mon brassinolide 24-epi"],
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
    "alpha-cypermethrin": ["alpha cypermethrin", "alpha xi permethrin", "permethrin", "cypermethrin"],
    "ametryn": ["ametrin", "ametryn"],
    "amino": ["amino"],
    "amisulbrom": ["amisulbrom"],
    "amonium-nitrate": ["amonium nitrat", "phot pho amoni"],
    "antibiotic": ["khang sinh"],
    "antibiotics": ["khang sinh"],
    "atrazine": ["atrazin"],
    "avermectin": ["avermectin"],
    "avermectin-b1a": ["avermectin b1a"],
    "avermectin-b1b": ["avermectin b1b"],
    "azol-450sc": ["azol 450sc"],
    "azoxystrobin": ["azoxistrobin", "azoxystrobin"],
    "azoxystrobincrop:rice": ["azoxystrobin lua"],
    "azoxytrobin": ["azoxistrobin", "azoxytrobin"],
    "bacillus-subtilis": ["bacillus subtilis", "vi khuan bacillus subtilis", "bacillus", "subtilis"],
    "badge-x2": ["badge x2"],
    "bentazone": ["bentazon", "bentazone"],
    "beta-cypermethrin": ["beta cypermethrin", "beta xi permethrin", "cypermethrin", "permethrin"],
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
    "cartap-hydrochloride": ["cartap hydroclorua", "thuoc diet sau cartap", "cartap", "hydrochloride"],
    "chloramphenicol": ["cloramphenicol", "thuoc khang sinh cloramphenicol"],
    "chlorfenapyr": ["clorfenapyr", "thuoc diet sau clorfenapyr", "chlorfenapyr"],
    "chlorothalonil": ["clorotalonil", "thuoc tri nam clorotalonil", "chlorothalonil"],
    "chlorothalonil-50sc": ["clorotalonil 50sc", "thuoc tri nam clorotalonil 50sc", "chlorothalonil"],
    "chlorothalonil-75wp": ["clorotalonil 75wp", "thuoc tri nam clorotalonil 75wp", "chlorothalonil"],
    "chlorpyrifos-ethyl": ["clorpyrifos etyl", "thuoc diet sau clorpyrifos etyl", "chlorpyrifos"],
    "chlorpyrifos-methyl": ["clorpyrifos metyl", "thuoc diet sau clorpyrifos metyl", "chlorpyrifos"],
    "contact-fungicide": ["thuoc diet nam tiep xuc", "thuoc tri nam tiep xuc"],
    "copper": ["chat dong", "dong"],
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
    "diquat-dibromide": ["thuoc diet co diquat", "thuoc diet co diquat dibromide"],
    "disease-control-chemical": ["thuoc diet benh", "thuoc phong tru benh"],
    "disease-fungicide-group-a": ["thuoc diet nam nhom a"],
    "disinfectant": ["thuoc khu trung", "thuoc tay trung"],
    "dithianon": ["thuoc diet nam dithianon", "dithianon"],
    "emamectin-benzoate": ["thuoc diet sau emamectin benzoate", "thuoc tru sau emamectin benzoate", "emamectin", "benzoate"],
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
    "gibberellic-acid": ["acid gibberellic", "chat kich thich tang truong gibberellic", "gibberellic"],
    "glufosinate-amonium": ["thuoc diet co glufosinate amonium", "glu", "glufosinate", "amonium"],
    "glufosinate-p": ["thuoc diet co glufosinate p"],
    "graduate-a": ["thuoc diet sau graduate a"],
    "growth-inhibitor": ["chat ngan can phat trien cay", "chat ngan can tang truong"],
    "haloxyfop-p-methyl": ["thuoc diet co haloxyfop p methyl", "haloxyfop"],
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
    "kresoxim-methyl": ["thuoc tru benh kresoxim methyl", "kresoxim"],
    "lactic-acid": ["axit lactic"],
    "lactofen": ["thuoc diet co lactofen", "lactofen"],
    "lambda-cyhalothrin": ["thuoc tru sau lambda cyhalothrin", "lambda", "cyhalothrin"],
    "lan-86": ["thuoc tru benh lan 86"],
    "legion": ["thuoc diet sau legion"],
    "liquid-fertilizer": ["phan bon duong luong", "phan bon loang"],
    "lufenuron": ["thuoc tru sau lufenuron", "lufenuron"],
    "lysol": ["chat tay trung lysol"],
    "mancozeb": ["thuoc tru benh mancozeb", "mancozeb"],
    "mefenoxam": ["thuoc tru benh mefenoxam"],
    "mesotrione": ["thuoc diet co mesotrione", "Mesotrione"],
    "meta-umizone": ["thuoc tru benh meta umizone"],
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
    "petroleum-oil": ["dau mo than", "dau mo than trong nong nghiep"],
    "phenthoate": ["phenthoate", "thuoc diet sau phenthoate"],
    "phosphite": ["phan photphit", "phan photphit trong nong nghiep"],
    "phosphonate": ["phan photphonat", "phan photphonat trong nong nghiep"],
    "phoxim": ["phoxim", "thuoc diet sau phoxim"],
    "phu-gia": ["chat phu gia", "chat phu gia trong nong nghiep"],
    "phytophthora-selective-medium": ["mo trung chon loc phytophthora", "mo trung phytophthora"],
    "pimaricin": ["khang sinh pimaricin", "pimaricin"],
    "pirimiphos-methyl": ["pirimiphos methyl", "thuoc diet sau pirimiphos methyl", "pirimiphos"],
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
    "prochloraz-manganese-complex": ["phuc hop prochloraz mangan"],
    "prochloraz-manganesse-complex": ["phuc hop prochloraz mangan", "prochloraz", "manganesse"],
    "propamocarb": ["thuoc tru nam propamocarb", "propamocarb"],
    "propamocarb-hcl": ["propamocarb hydrochloride", "propamocarb", "hydrochloride"],
    "propiconazole": ["thuoc tru nam propiconazole"],
    "propoxur": ["thuoc tru sau propoxur", "propoxur"],
    "proteose-peptone": ["proteose pepton"],
    "pymetrozine": ["thuoc tru sau pymetrozine", "pymetrozine"],
    "pyraclostrobin": ["thuoc tru nam pyraclostrobin", "pyraclostrobin"],
    "pyrethroid": ["phan bo pyrethroid", "thuoc tru sau pyrethroid"],
    "pyridaben": ["thuoc tru sau pyridaben", "pyridaben"],
    "pyriproxyfen": ["thuoc tru sau pyriproxyfen", "pyriproxyfen"],
    "quadris-top": ["thuoc tru nam quadris top"],
    "quicklime": ["voi toi"],
    "quintozene": ["thuoc tru nam quintozene"],
    "quizalofop-p-ethyl": ["quizalofop p ethyl", "quizalofop"],
    "r333": ["thuoc tru sau r333"],
    "ridomil": ["thuoc tru nam ridomil"],
    "rose-bengal": ["thuoc nhuom rose bengal"],
    "s-metolachlor": ["thuoc tru sau s metolachlor", "metolachlor"],
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
    "terbuthylazine": ["thuoc diet co terbuthylazine"],
    "terrachlor": ["thuoc diet co terrachlor"],
    "than-duoc-sach-benh": ["than duoc sach benh", "thuoc sach benh"],
    "thiabendazole": ["thuoc thiabendazole", "thuoc tri nam thiabendazole"],
    "thiacloprid": ["thuoc diet sau thiacloprid", "thiacloprid"],
    "thiamethoxam": ["thuoc diet sau thiamethoxam", "thiamethoxam"],
    "thiosultap-sodium": ["thuoc diet sau thiosultap sodium", "thiosultap"],
    "thiram": ["thuoc thiram", "thuoc tri nam thiram", "thiram"],
    "thuoc-tru-nam": ["thuoc diet nam", "thuoc tri nam"],
    "thuoc-tru-nhen": ["thuoc diet nhen", "thuoc tru nhen"],
    "tilt": ["thuoc tilt", "thuoc tri nam tilt"],
    "tolfenpyrad": ["thuoc diet sau tolfenpyrad", "thuoc tolfenpyrad", "tolfenpyrad"],
    "topramezone": ["thuoc diet co topramezone", "thuoc topramezone", "topramezone"],
    "toxic-chemical": ["chat doc", "chat doc hai"],
    "triadimefon": ["thuoc tri nam triadimefon", "thuoc triadimefon", "triadimefon"],
    "tricyclazole": ["thuoc tri nam tricyclazole", "thuoc tricyclazole"],
    "tricylazole": ["thuoc tri nam tricylazole", "thuoc tricylazole", "tricylazole"],
    "trifloxystrobin": ["thuoc tri nam trifloxystrobin", "thuoc trifloxystrobin"],
    "trinong-50wp": ["thuoc tri nam trinong 50wp", "thuoc trinong 50wp"],
    "trioxystrobin": ["thuoc tri nam trioxystrobin", "thuoc trioxystrobin"],
    "ultra-flourish": ["thuoc phan bon ultra flourish", "thuoc ultra flourish"],
    "wa-0-05": ["thuoc wa 0 05"],
    "zearalenone": ["chat doc zearalenon", "chat doc zearalenone"],
    "zhongshengmycin": ["thuoc khang sinh zhongshengmycin", "thuoc zhongshengmycin", "zhongshengmycin"],
}

CROP_ALIASES = {
    "agricultural-products": ["san pham nong nghiep", "san pham nong san"],
    "apple": ["cay tao", "tao"],
    "avocado": ["qua bo", "cay bo", "trai bo"],
    "banana": ["cay chuoi", "chuoi"],
    "bap": ["bap", "cay bap", "corn", "ngo"],
    "bap-cai": ["bap cai", "cay bap cai"],
    "barley": ["cay lua mi", "lua mi"],
    "bean": ["cay dau", "dau"],
    "bitter-melon": ["kho qua", "qua kho qua"],
    "black-pepper": ["tieu", "tieu den"],
    "bok-choy": ["cai thieu", "cai thieu xanh"],
    "bottle-gourd": ["bi dao", "bi huong"],
    "brassicaceae": ["ho ca cai", "ho cai"],
    "broadleaf-crops": ["cay la rong"],
    "buoi": ["buoi"],
    "ca-chua": ["ca chua", "ca chua bi"],
    "ca-phe": ["ca phe"],
    "ca-tim": ["ca tim"],
    "cabbage": ["bap cai", "cai bap"],
    "cacao": ["ca cao"],
    "cam": ["cam"],
    "cay-an-qua": ["cam", "quyt", "buoi", "xoai", "cafe", "ca phe"],
    "cam-quyt": ["cam quyt"],
    "cao-su": ["cao su"],
    "carrot": ["ca rot"],
    "cashew": ["dieu", "hat dieu"],
    "cassava": ["khoai mia", "khoai san"],
    "cay-an-trai": ["cay an trai"],
    "cay-cam": ["cay cam"],
    "cay-con": ["cay con"],
    "cay-trong": ["cay trong"],
    "cay-truong-thanh": ["cay truong thanh"],
    "central-vietnam-crops": ["cay trung bo"],
    "cereal-crops": ["cay lua"],
    "cereal-grains": ["hat lua"],
    "chanh": ["chanh"],
    "che": ["che"],
    "chewing-insect-pests": ["sau can", "sau can la"],
    "chili": ["ca ot", "ot"],
    "chili-pepper": ["ca ot", "ot"],
    "chrysanthemum": ["cu cuc"],
    "citrus": ["cay ho cam", "ho cam"],
    "cocoa": ["ca cao"],
    "coffee": ["ca phe"],
    "common-bean": ["dau", "dau chung"],
    "copper-sensitive-crops": ["cay nhay cam dong"],
    "corn": ["bap ngo", "ngo", "bap"],
    "cotton": ["bo vai"],
    "crop": ["cay trong"],
    "crop-disease": ["benh cay", "benh cay trong"],
    "crop-health": ["suc khoe cay trong", "tinh trang cay trong"],
    "crop-management": ["ky thuat trong cay", "quan ly cay trong"],
    "crop-plant": ["cay trong", "cay trong nong nghiep"],
    "crop-production": ["san luong nong nghiep", "san xuat nong nghiep"],
    "crop-protection": ["bao ve cay trong", "phong tru sau benh"],
    "crop-yield": ["nang suat cay trong", "san luong cay trong"],
    "cruciferous": ["ho ca cai", "ho cai"],
    "cruciferous-crops": ["cay trong ho cai", "rau ho cai"],
    "cu-dau": ["cu dau", "cu dau nong nghiep"],
    "cucumber": ["cay dua chuot", "dua chuot"],
    "cucurbitaceae": ["ho bau bi", "ho bau bi rau qua"],
    "dau-nanh": ["dau nanh", "hat dau nanh"],
    "dau-phong": ["dau phong", "hat dau phong"],
    "dau-tay": ["dau tay", "hat dau tay"],
    "dau-tuong": ["dau tuong", "hat dau tuong"],
    "dau-xanh": ["dau xanh", "hat dau xanh"],
    "dieu": ["cay dieu", "hat dieu"],
    "dragon-fruit": ["cay thanh long", "thanh long"],
    "dry-field": ["cay mua kho", "lua mua kho"],
    "du-du": ["cay du du", "du du"],
    "dua": ["cay dua", "dua"],
    "durian": ["cay sau rieng", "sau rieng"],
    "early-stage-crop": ["cay con non", "giai doan dau cay"],
    "edible-mushroom": ["nam an duoc", "nam an duoc trong"],
    "edible-mushrooms": ["cac loai nam an duoc", "nam an duoc"],
    "eggplant": ["ca tim", "cay ca tim"],
    "field": ["canh dong", "ruong"],
    "field-crop": ["cay cong nghiep", "cay nong nghiep"],
    "field-crops": ["cac loai cay nong nghiep", "cay nong nghiep"],
    "flowering-crop": ["cay dang ra hoa", "cay ra hoa"],
    "flowering-plants": ["cay dang ra hoa", "cay ra hoa"],
    "flowering-stage": ["giai doan ra hoa", "thoi ky ra hoa"],
    "flowers": ["cay hoa", "hoa"],
    "general": ["cay trong chung"],
    "general-crop": ["cay trong chung"],
    "general-crops": ["cay trong chung"],
    "ginger": ["gung"],
    "grape": ["nho"],
    "grapefruit": ["buoi"],
    "grapevine": ["cay nho"],
    "green-bean": ["dau xanh"],
    "green-beans": ["dau xanh"],
    "green-cauliflower": ["sup lo xanh"],
    "green-mung-bean": ["dau xanh"],
    "green-onion": ["hanh la"],
    "greenhouse-crop": ["cay trong nha kinh"],
    "gung": ["gung"],
    "hanh": ["hanh"],
    "hanh-la": ["hanh la"],
    "hoa": ["hoa cuc", "hoa dong tien", "hoa hong"],
    "harvest-stage": ["giai doan thu hoach"],
    "healthy-crop-area": ["dien tich cay trong khoe manh"],
    "healthy-seedlings": ["moc giong khoe manh"],
    "ho-tieu": ["ho tieu"],
    "hoa-dao": ["hoa dao"],
    "hoa-hong": ["hoa hong"],
    "hoa-hong-dai-multiflora": ["hoa hong dai"],
    "hoa-ly": ["hoa ly"],
    "hoa-mai": ["hoa mai"],
    "industrial-crop": ["cay cong nghiep"],
    "industrial-crops": ["cay cong nghiep"],
    "jackfruit": ["mit"],
    "jicama": ["cu su"],
    "jujube": ["tao ta"],
    "kenaf": ["bam"],
    "khoai-lang": ["khoai lang"],
    "khoai-tay": ["khoai tay"],
    "khom": ["dua hau"],
    "lac": ["dau phong"],
    "leaf": ["la cay"],
    "legume": ["cay dau", "dau"],
    "legumes": ["cay dau"],
    "lemon": ["chanh vang"],
    "lettuce": ["rau xon"],
    "lime": ["chanh tay"],
    "litchi": ["vai"],
    "longan": ["nhan"],
    "lotus": ["sen"],
    "lua": ["lua"],
    "lychee": ["vai"],
    "mai": ["hoa mai"],
    "mai-flower": ["hoa mai"],
    "maize": ["ngo"],
    "mandarin": ["quyt"],
    "mang-cau": ["mang cau"],
    "mango": ["xoai"],
    "me-vung": ["me vung"],
    "mia": ["mia"],
    "millet": ["ke"],
    "mit": ["mit"],
    "mulberry": ["dau"],
    "multiple-crops": ["nhieu loai cay trong"],
    "mung-bean": ["dau xanh"],
    "mup-bitter-gourd": ["muop dang"],
    "narrow-leaf-crops": ["cay la hep"],
    "nematode-resistant-crops": ["cay chong nematode"],
    "ngo": ["ngo"],
    "nhieu-loai-cay-trong": ["nhieu loai cay trong"],
    "nho": ["nho"],
    "non-selective": ["khong phan biet"],
    "nonbearing-citrus": ["cay cam chua cho trai"],
    "nursery-plants": ["cay uom"],
    "nut-trees": ["cay hat"],
    "onion": ["hanh"],
    "orange": ["cam"],
    "orchard": ["vuon cay"],
    "organic-farming": ["canh tac huu co"],
    "ot": ["ot"],
    "pak-choi": ["cai bap"],
    "pea": ["dau ha lan"],
    "peach": ["cay dao", "dao"],
    "peanut": ["dau phong", "hat dau phong"],
    "pear": ["cay le", "le"],
    "pepper": ["cay tieu", "tieu"],
    "pineapple": ["dua", "dua hau", "dua tay"],
    "plum": ["cay man", "man"],
    "potato": ["khoai", "khoai tay"],
    "potted-plants": ["cay trong chau", "cay trong trong chau"],
    "pre-harvest": ["thoi ky truoc thu hoach", "truoc thu hoach"],
    "quyt": ["cay quyt", "quyt"],
    "quyt-satsuma": ["quyt satsuma", "quyt satsuma nhat"],
    "ra-hoa": ["ra hoa", "thoi ky ra hoa"],
    "rau": ["rau", "rau an","bap cai", "dau tuong", "cai thia", "bap", "ngo", "sup lo", "xu hao", "ca chua", "hanh", "muop"],
    "rau-cai": ["rau cai", "rau cai la"],
    "rau-muong": ["rau muong"],
    "rau-mau": ["rau mau", "rau mau an"],
    "resistant-variety": ["giong chong benh", "giong khang benh"],
    "rice": ["cay lua", "lua"],
    "rice-husk": ["vo lua", "vo lua sau khi tach"],
    "roi-man": ["cay roi man", "roi man"],
    "root": ["bo re", "re cay"],
    "root-crop": ["cay goc", "cay goc trong"],
    "root-crops": ["cay goc", "cay goc trong"],
    "root-zone": ["vung bo re", "vung re"],
    "rose": ["cay hoa hong", "hoa hong"],
    "rose-bush": ["buom hoa hong", "buom hong"],
    "rubber": ["cao su", "cay cao su"],
    "rubber-tree": ["cay cao su"],
    "same-variety": ["cung giong", "giong giong nhau"],
    "san": ["cay san"],
    "sapodilla": ["cay hong xi", "hong xi"],
    "satsuma-mandarin": ["quyt satsuma", "quyt satsuma nhat"],
    "sau-rieng": ["cay sau rieng", "sau rieng"],
    "seed": ["hat giong", "hat trong"],
    "seedlings": ["cay con", "giong cay con"],
    "sensitive-plants": ["cay nhay cam"],
    "sesame": ["hat va"],
    "short-term-crops": ["cay vu dai han ngan", "cay vu ngan han"],
    "soil": ["dat trong cay"],
    "solanaceae": ["ho ca"],
    "sorghum": ["ngo mi"],
    "soursop": ["man", "qua man"],
    "soybean": ["dau nanh", "hat dau nanh"],
    "squash": ["bi ngo", "qua bi ngo"],
    "strawberry": ["dau tay", "qua dau tay"],
    "sugarcane": ["cay mia", "mia"],
    "sweet-orange": ["cam ngot"],
    "sweet-potato": ["khoai lang"],
    "tangerine": ["quyt"],
    "taro": ["khoai mon"],
    "tea": ["cay tra", "tra"],
    "thanh-long": ["qua thanh long", "thanh long"],
    "thuoc-la": ["cay thuoc la", "thuoc la"],
    "tobacco": ["cay thuoc la", "thuoc la"],
    "tomato": ["ca chua", "qua ca chua"],
    "upland-crops": ["cay vu nui"],
    "various-crops": ["cac loai cay trong", "cay trong da dang"],
    "vegetable": ["rau an"],
    "vegetable-crops": ["cay rau"],
    "vegetables": ["rau an"],
    "vuon-cay-an-trai": ["vuon cay an trai"],
    "watermelon": ["dua hau", "qua dua hau"],
    "weed": ["co dai", "mac-co", "rau sam", "co cuc", "cho de", "den gai", "co chan vit", "co long vuc", "co man trau"],
    "weed-control": ["diet co dai", "kiem soat co dai"],
    "wheat": ["lua mi"],
    "woody-plants": ["cay go"],
    "xoai": ["qua xoai", "xoai"],
    "young-plant": ["cay con"],
    "young-plants": ["cay con"],
}

DISEASE_ALIASES = {
    "nhom-a": ["nhom benh a", "nhom a", "nam nhom a", "nhom nam a", "benh nhom a", "than thu", "dao on", "dom vong", "dom tim", "bi thoi", "chay la", "dom nau", "dom la", "heo ru", "chet cham", "chay day", "thoi re", "lua von", "lem lep hat", "phan trang", "moc xam", "nam long chuot", "ghe la", "ghe trai", "dom den", "thoi than", "thoi hach", "thoi re", "benh thoi canh", "chay canh", "thoi qua", "benh chet canh", "benh scab", "benh ghe", "san vo", "tiem lua", "vang be", "thoi trai", "kho dot", "chet canh", "nut than", "chay nhua", "benh dom nau", "kho"
],
    "nhom-b": ["nhom benh b", "nhom b", "nam nhom b", "nhom nam b", "benh nhom b", "lo co re", "heo cay con", "chay la", "kho van", "nam hong", "heo ru", "moc trang", "co re bi thoi nau", "thoi nau", "thoi nhun", "benh chet rap cay con", "thoi trai", "thoi than", "ri sat", "than hat lua", "benh ri sat dau tuong", "than thu", "dom la lon", "lem lep hat"
],
    "nhom-o": ["nhom benh o", "nhom o", "nam nhom o", "nhom nam o", "benh nhom o", "suong mai", "benh thoi re", "thoi ngon", "thoi mam", "chet nhanh", "thoi trai", "nut than", "xi mu", "vang la", "chet than", "chet canh", "thoi re", "chet cay con", "moc suong", "gia suong mai", "soc trang", "bach tang", "moc xuong", "ri trang", "nam trang", "phong trang"],
}

PEST_ALIASES = {
    "aflatoxin": ["doc to aflatoxin"],
    "airborne-pathogen": ["tac nhan gay benh bay qua khong khi"],
    "albuginaceae": ["nam bao tu"],
    "alkalinity": ["do kiem dat"],
    "alternaria-spp": ["nam alternaria"],
    "annual-sedge": ["co nam"],
    "annual-weed": ["co nam"],
    "ant": ["kien"],
    "aphid": ["ray mem"],
    "aphids": ["ray mem"],
    "armillaria-fungi": ["nam vang chan"],
    "armyworm": ["sau do"],
    "ascospore": ["bap tu"],
    "avoid": ["tranh"],
    "bacteria": ["vi khuan"],
    "bacterial-disease": ["benh vi khuan"],
    "bacterial-diseases": ["benh vi khuan"],
    "bacterial-pathogen": ["tac nhan vi khuan"],
    "bao-tu": ["bao tu"],
    "bao-tu-hau": ["bao tu hau"],
    "bao-tu-suong-mai-gia": ["bao tu suong mai gia"],
    "beneficial-insects": ["con trung co ich", "con trung huu ich"],
    "benh-thoi-re": ["benh thoi re", "benh thoi re cay trong"],
    "benh-thoi-vo": ["benh thoi vo", "benh thoi vo hat"],
    "bo-canh-to": ["bo canh to", "bo canh to tren cay"],
    "bo-ngau": ["bo ngau", "bo ngau tren cay"],
    "bo-nhay": ["bo nhay", "bo nhay tren cay"],
    "bo-phan": ["bo phan", "bo phan tren cay"],
    "bo-phan-trang": ["bo phan trang", "bo phan trang tren cay"],
    "bo-tri": ["bo tri", "xu ly tri", "xu ly bo tri", "bo hut", "bo tri tren cay"],
    "bo-xit": ["bo xit", "bo xit tren cay", "bo xit muoi"],
    "bo-ha": ["bo ha", "sung dat", "sung"],
    "borer": ["sau duc than", "sau duc trong"],
    "borers": ["sau duc than", "sau duc trong"],
    "broad-leaf-weed": ["co la rong", "co la rong trong ruong"],
    "broad-leaf-weeds": ["co la rong", "co la rong trong ruong"],
    "broadleaf-weed": ["co la rong", "co la rong trong ruong"],
    "broadleaf-weeds": ["co la rong", "co la rong trong ruong"],
    "co-chao": ["co chao", "co chao trong ruong"],
    "co-chi": ["co chi", "co chi trong ruong"],
    "co-dai-la-hep": ["co dai la hep", "co dai la nho", "la hep"],
    "co-dai-la-rong": ["co dai la rong", "co dai la to", "la rong"],
    "co-duoi-phung": ["co duoi phung", "co duoi phung la"],
    "co-man-trau": ["co man trau", "co man trau la", "man-trau"],
    "co-tranh": ["co tranh", "co tranh la"],
    "coffee-mealybug": ["bo trung ban ca phe", "ruoi ban ca phe"],
    "cold-humid-climate": ["dieu kien lanh am", "khi hau lanh am"],
    "compacted-soil": ["dat nen cat cung", "dat nen cung"],
    "coniothyrium-spp": ["nam benh coniothyrium", "nam coniothyrium"],
    "contaminants": ["chat ban", "chat gay o nhiem"],
    "copper-ion": ["dong ion", "ion dong"],
    "copper-residue": ["du luong dong", "ton dong tren cay"],
    "copper-toxicity": ["doc dong", "nguy hiem do dong"],
    "corn-earworm": ["sau bua", "sau bua bap"],
    "cyperus-rotundus": ["co co", "co co tron"],
    "dao-on": ["benh dao on", "dao on"],
    "deep-leaf-folder": ["sau gap la", "sau gap la sau"],
    "disease-control": ["kiem soat benh", "phong chong benh"],
    "disease-prevention": ["phong benh", "phong ngua benh"],
    "disease-spot": ["vet benh", "vet benh tren la"],
    "doi-duc-la": ["benh doi duc la", "doi duc la"],
    "economic-impact": ["anh huong kinh te", "tac dong kinh te"],
    "eggs": ["trung con trung", "trung sau"],
    "fruit-borer": ["sau duc qua", "sau duc trai"],
    "fruit-damage": ["sau hai qua", "sau hai trai"],
    "fruit-fly": ["ruoi hai qua", "ruoi hai trai"],
    "fungal-contamination": ["nam nhiem ban", "nam nhiem benh"],
    "fungal-disease": ["benh nam", "benh nam cay trong"],
    "fungal-diseases": ["cac benh nam", "cac benh nam cay trong"],
    "fungal-diseases-group-a": ["nhom benh nam a"],
    "fungal-infection": ["nam nhiem", "nam nhiem benh"],
    "fungal-nematodes": ["nam nematod", "nam nematode"],
    "fungal-pathogen": ["tac nhan gay benh nam", "tac nhan nam"],
    "fungal-spores": ["bap nam", "bap nam benh"],
    "fungi": ["nam", "vi nam"],
    "fungicide": ["thuoc diet nam", "thuoc diet nam benh"],
    "furry-spider-mite": ["ruoi ve long", "ruoi ve long benh"],
    "fusarium": ["benh nam fusarium", "nam fusarium"],
    "ghe-seo": ["benh ghe seo", "benh ghe seo cay trong"],
    "glyphosate-resistant-weeds": ["cac loai co khang glyphosate", "co khang", "khang glyphosate"],
    "golden-apple-snail": ["oc buou", "oc buou vang", "oc vang"],
    "golden-snail": ["oc vang", "oc vang vang"],
    "green-bug": ["ran la xanh", "ran lua xanh", "ran xanh"],
    "green-leaf-bug": ["ran la", "ran la xanh", "ran xanh tren la"],
    "insect": ["con trung", "sau benh"],
    "insect-control": ["diet con trung", "kiem soat con trung", "phong tru con trung"],
    "insect-larvae": ["au con trung", "au trung", "ot con trung"],
    "insect-pests": ["con trung gay hai", "sau benh"],
    "insect-vector": ["con trung mang benh", "con trung truyen benh", "con trung truyen virus"],
    "insect-vectors": ["con trung mang benh", "con trung truyen benh", "con trung truyen virus"],
    "insects": ["con trung", "sau benh"],
    "invasive-species": ["loai xam lan", "sinh vat xam lan", "sinh vat xam lan trong nong nghiep"],
    "khang-thuoc": ["khang thuoc", "khang thuoc con trung", "khang thuoc sau benh"],
    "kien": ["con kien", "kien", "kien trong nong nghiep"],
    "larvae": ["au", "au trung", "ot con trung"],
    "lay-benh": ["lay benh", "lay nhiem benh", "truyen benh"],
    "leaf-beetle": ["bo an la", "bo la", "bo la cay"],
    "leaf-burn": ["la bi chay", "la bi dot", "la bi hong do nong do"],
    "leaf-chewing-insects": ["con trung an la", "con trung gay hai la", "sau an la"],
    "leaf-damage": ["gay hai tren la", "thiet hai la", "thiet hai tren la"],
    "leaf-eating-caterpillar": ["sau an la", "sau an la cay"],
    "leaf-eating-insect": ["coc an la", "coc an la cay"],
    "leaf-eating-insects": ["coc an la", "coc an la cay"],
    "leaf-fall": ["la roi", "roi la"],
    "leaf-folder": ["sau gap la", "sau gap la cay"],
    "leaf-folder-larvae": ["ot sau gap la", "ot sau gap la cay"],
    "leaf-miner": ["sau an trong la", "sau an trong la cay"],
    "leaf-miner-fly": ["ruoi an trong la", "ruoi an trong la cay"],
    "leaf-miners": ["sau an trong la", "sau an trong la cay"],
    "leaf-roller": ["sau cuon la", "sau cuon la cay"],
    "leaf-spots": ["dot tren la", "vet dot tren la"],
    "leafhopper": ["bo tron la", "bo tron la cay"],
    "microorganisms": ["vi sinh vat", "vi sinh vat gay hai"],
    "mite": ["bo trung", "bo trung cay trong"],
    "mites": ["bo trung", "bo trung cay trong"],
    "mo": ["mo", "mo cay trong"],
    "moc": ["nam moc", "nam moc cay trong"],
    "mold": ["nam moc", "nam moc cay trong"],
    "mold-spores": ["bao tu nam moc", "bao tu nam moc cay trong"],
    "mosquito": ["muoi", "muoi cay trong"],
    "mosquito-bug": ["muoi", "muoi cay trong"],
    "mot-duc-canh": ["mot duc canh", "duc canh", "mot duc canh cay trong"],
    "motile-spores": ["bao tu co dong", "bao tu co dong cay trong"],
    "muoi-hanh": ["muoi hanh", "muoi hanh cay trong"],
    "mycotoxin": ["doc to nam", "doc to nam moc"],
    "nam": ["nam", "nam cay trong"],
    "nam-benh": ["nam benh", "nam benh cay trong"],
    "nam-fusarium": ["nam benh fusarium", "nam fusarium"],
    "nam-phytophthora": ["nam benh phytophthora", "nam phytophthora"],
    "narrow-leaf-weed": ["co hep la", "co hep la cay trong"],
    "narrow-leaf-weeds": ["co hep la", "co hep la cay trong"],
    "narrowleaf-weed": ["co hep la", "co hep la cay trong"],
    "narrowleaf-weeds": ["co hep la", "co hep la cay trong"],
    "nematode": ["rut gon", "rut gon cay trong"],
    "nematodes": ["rut gon", "rut gon cay trong"],
    "nhen": ["con nhen", "ruoi nhen", "nhen"],
    "nhen-do": ["nhen do", "ruoi nhen do"],
    "nhen-khang-thuoc": ["nhen khang thuoc", "ruoi nhen khang thuoc"],
    "nhen-long-nhung": ["nhen long nhung", "ruoi nhen long nhung"],
    "nhen-trang": ["nhen trang", "ruoi nhen trang"],
    "nhen-vang": ["nhen vang", "ruoi nhen vang"],
    "oc-buu-vang": ["oc buu vang", "oc vang","oc"],
    "pesticide": ["thuoc bao ve thuc vat", "thuoc tru sau"],
    "pests": ["sau benh", "sau hai"],
    "ph": ["do axit kiem", "do ph"],
    "phan-trang": ["benh phan trang", "phan trang"],
    "rape-sap": ["sau hut mach", "sau hut mach cay"],
    "rat": ["chuot", "chuot hai cay"],
    "ray": ["ray", "ray tren cay"],
    "ray-bong": ["ray bong", "ray bong tren cay"],
    "ray-lung-trang": ["ray lung trang", "ray lung trang tren cay"],
    "ray-mem": ["ray mem", "ray mem tren cay"],
    "ray-nau": ["ray nau", "ray lung trang", "ray nau tren cay"],
    "ray-phan": ["ray phan", "ray phan tren cay"],
    "ray-xanh": ["ray xanh", "ray-chong-canh", "ray phan", "ray mem", "bo phan trang", "bo xit"],
    "red-mite": ["ruoi do", "ruoi do tren cay"],
    "red-spider-mite": ["nhen gie", "nhen do", "nhen long nhung", "nhen trang"],
    "rep": ["rep", "rep tren cay"],
    "rep-bong-xo": ["rep bong xo", "rep bong xo tren cay"],
    "rep-mem": ["rep mem", "rep mem tren cay"],
    "rep-muoi": ["rep muoi", "rep muoi tren cay"],
    "rep-sap": ["rep sap", "rep vay", "rep sap tren cay"],
    "rep-vay": ["rep vay", "rep sap", "rep vay tren cay"],
    "resistant-weed": ["co khang thuoc", "co khang thuoc tru sau"],
    "resistant-weeds": ["co khang thuoc diet co", "co khang thuoc tru sau"],
    "rice-bug": ["sau com", "sau ruong"],
    "rice-mite": ["ngua lua", "ngua lua lua"],
    "rice-spider-mite": ["ngua nhan", "ngua nhan lua"],
    "rice-stink-bug": ["sau thoi", "sau thoi lua"],
    "rodent": ["chuot", "chuot lua", "chuot ruong"],
    "root-knot-nematode": ["ruot tron gay benh re", "ruot tron gay hai"],
    "root-parasite": ["ky sinh re", "thuc vat ky sinh re"],
    "root-rot": ["benh thoi re", "thoi re"],
    "ruoi-vang": ["ruoi vang", "ruoi vang lua"],
    "rust-fungus": ["benh nam giang", "nam giang"],
    "sau-benh": ["sau benh", "sau gay benh"],
    "sau-bo": ["sau bo", "sau bo la"],
    "sau-chich-hut": ["sau chich hut", "sau hut mau"],
    "sau-cuon-la": ["sau cuon la", "sau cuon la lua"],
    "sau-dat": ["sau dat", "sau dat lua"],
    "sau-duc-be": ["sau duc be", "sau duc than be"],
    "sau-duc-canh": ["sau duc canh", "sau duc canh lua"],
    "sau-duc-qua": ["sau duc qua", "sau duc qua lua"],
    "sau-duc-than": ["sau duc than", "sau duc than lua"],
    "sau-duc-trong": ["sau duc trong", "sau duc trong lua"],
    "sau-hai": ["sau hai", "sau hai lua"],
    "sau-hanh": ["sau hanh", "sau hanh lua"],
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
    "snail": ["oc sen", "oc sen trong ruong"],
    "soil-fungi": ["nam dat", "nam gay benh tren dat"],
    "soil-fungus": ["nam dat", "nam gay benh tren dat"],
    "soil-insect": ["sau dat", "sau trong dat"],
    "soil-insects": ["sau dat", "sau trong dat"],
    "soil-moisture": ["do am dat", "do am dat trong ruong"],
    "soil-pathogen": ["benh gay boi tac nhan tren dat", "tac nhan gay benh tren dat"],
    "soil-pest": ["sau benh dat", "sau benh tren dat"],
    "soil-pests": ["sau benh dat", "sau benh tren dat"],
    "spider": ["con nhen", "nhen"],
    "spider-mite": ["bo trung nhen", "bo trung nhen cay"],
    "spore": ["bao tu nam", "bao tu nam benh"],
    "spores": ["bao tu nam", "bao tu nam benh"],
    "spot": ["vet den", "vet den tren la"],
    "stem-borer": ["sau duc cot", "sau duc than"],
    "stem-crack": ["nut cot", "nut than"],
    "stress-cay": ["cay bi stress", "stress o cay", "stress"],
    "stress-cay-trong": ["cay trong bi stress", "stress cay trong", "stress"],
    "sucking-pest": ["sau hut", "sau hut mau"],
    "sung-khoai": ["sung khoai", "sung khoai tay"],
    "suong-mai": ["benh suong mai", "suong mai"],
    "than-thu": ["sau than", "than thu"],
    "thrips": ["bo tri", "tri tren dua non", "tri tren la"],
    "thuoc-bvtv": ["thuoc bao ve thuc vat", "thuoc tru sau"],
    "ve-sau": ["con ve sau", "ve sau"],
    "weeds": ["co dai", "co tranh", "co dai trong lua", "co dai trong ruong"],
    "wet-soil": ["dat am", "dat am uot", "dat uot"],
    "white-mite": ["benh ran trang", "ran trang tren cay"],
    "white-mold": ["benh nam trang", "nam trang"],

}

PRODUCT_ALIASES = {
    "afenzole-top-325sc": ["thuoc diet sau afenzole top", "thuoc tru sau afenzole", "afenzoletop"],
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
    "gone-super-350ec": ["thuoc diet sau gone super", "thuoc tru sau gone super 350ec", "gon super", "gone super"],
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
    "trum-nam-benh": ["thuoc trum nam benh", "trum nam benh", "tru nam", "nam benh"],
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
    "chessin-600wp": ["thuoc diet sau chessin 600wp", "chessin"],
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
    "emycin-4wp": ["thuoc emycin 4wp", "emycin"],
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
    "fuji-boss-30sc": ["thuoc fuji boss 30sc", "fujiboss", "fuji boss", "boss", "fuji"],
    "fujiboss-30sc": ["thuoc fuji boss 30sc", "fujiboss", "fuji boss", "boss", "fuji"],
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
    "khongray-54wp": ["thuoc bao ve thuc vat khongray 54wp", "thuoc khongray 54wp", "khongray", "khongray-54wp"],
    "king-cide-japan-460sc": ["thuoc bao ve thuc vat king cide japan 460sc", "thuoc king cide japan 460sc", "king cide japan 460sc", "king cide japan", "king cide", "kingcide", "king-cide"],
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
    "mitop-one-390sc": ["mitop-one", "mitop one", "mitop", "thuoc mitop one 390sc"],
    "modusa-960ec": ["modusa", "modusa 960 ec", "thuoc modusa 960 ec"],
    "modusa-960ec-gold": ["modusa", "modusa 960 ec gold", "thuoc modusa 960 ec gold"],
    "nakano-50wp": ["nakano", "nakano 50 wp", "thuoc nakano 50 wp"],
    "napoleon-fortazeb-72wb": ["napoleon", "fortazeb", "napoleon-fortazeb", "napoleon fortazeb 72 wb", "thuoc napoleon fortazeb 72 wb"],
    "naticur": ["naticur", "thuoc naticur"],
    "nekko-69wp": ["nekko 69 wp", "thuoc nekko 69 wp", "nekko"],
    "newfosinate-150sl": ["newfosinate 150 sl", "thuoc newfosinate 150 sl", "newfosinate"],
    "nhen-kim-cuong": ["kim cuong", "thuoc diet nhen kim cuong"],
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
    "probicol-200wp": ["thuoc probicol", "thuoc probicol 200wp", "probicol", "propicol"],
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
    "sieu-diet-chuot": ["thuoc diet chuot", "thuoc sieu diet chuot", "diet chuot", "diet chuot","chuot"],
    "sieu-diet-mam": ["thuoc diet mam", "thuoc sieu diet mam", "diet mam"],
    "sieu-diet-nhen": ["thuoc diet nhen", "thuoc sieu diet nhen", "diet nhen"],
    "sieu-diet-sau": ["thuoc diet sau", "thuoc sieu diet sau", "diet-sau"],
    "sinapy-ram": ["thuoc sinapy", "thuoc sinapy ram", "sinapy", "sinapyram", "ram"],
    "sinapyram-80wg": ["thuoc sinapyram 80wg", "sinapy", "sinapyram", "ram"],
    "somethrin-10ec": ["thuoc somethrin", "thuoc somethrin 10ec", "somethrin"],
    "su-tu-do": ["thuoc su tu do", "su tu do"],
    "suparep-22-4sc": ["thuoc suparep 22 4sc", "thuoc suparep", "suparep"],
    "suparep-400wp": ["thuoc suparep 400wp", "suparep"],
    "supermario-70sc": ["thuoc supermario 70sc", "supermario"],
    "suria-10gr": ["thuoc suria 10gr", "suria"],
    "suron-800wp": ["thuoc suron 800wp", "suron"],
    "takiwa-22sc": ["thuoc takiwa 22sc", "takiwa"],
    "tamatras": ["thuoc tamatras", "tamatras", "spirotetramat tolfenpyrad", "spirotetramat", "tolfenpyrad"],
    "tamiko-50ec": ["thuoc tamiko 50ec", "tamiko"],
    "tatsu-25wp": ["thuoc tatsu 25wp", "tatsu"],
    "tembo-8od-vua-co-ngo": ["thuoc tembo 8od vua co ngo", "tembo", "vua co ngo", "8od"],
    "thalonil-75wp": ["thuoc thalonil 75wp", "thalonil"],
    "thuoc-bilu": ["thuoc bilu"],
    "thuoc-chuot-forwarat-0-005-wax-block": ["thuoc chuot forwarat 0 005 wax block", "chuot"],
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
    "vam-co-dktazole-480sl": ["thuoc vam co dktazole 480sl", "vam co", "dktazole", "vamco"],
    "vamco-480sl": ["thuoc vamco 480sl", "vam co", "vamco"],
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
    "luu-dan-manh": ["luu dan manh", "luu dan nao manh"],
    "luu-dan": ["luu dan", "lu dan", "luu dan nao", "lu dan nao"],
    "tiep-xuc-manh": ["tiep xuc manh", "tiep xuc nào manh"],
    "tiep-xuc": ["tiep xuc", "tiep suc", "tiep xuc nao", "tiep suc nao"],
    "tiep-xuc-luu-dan-manh": ["tiep xuc va luu dan manh", " tiep xuc luu dan manh", "tiep xuc, luu dan manh", "tiep xuc va luu dan nao manh", " tiep xuc luu dan nao manh", "tiep xuc, luu dan manh", "luu dan manh, tiep xuc", "tiep xuc + luu dan manh", "luu dan manh + tiep xuc", "manh"],
    "tiep-xuc-luu-dan": ["tiep xuc va luu dan", "tiep xuc luu dan", "tiep xuc va luu dan nao", "tiep xuc luu dan nao", "tiep xuc, luu dan", "luu dan, tiep xuc", "tiep xuc + luu dan", "luu dan + tiep xuc nao"],
    "xong-hoi-manh": ["xong hoi manh", "xong hoi nao manh"],
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


# ======================
# 3) GENERIC EXTRACTOR (rút gọn – thay cho extract_chemicals/extract_pests/...)
#    - Không tối ưu AC ở đây để tập trung vào "không lặp rules"
#    - Khi cần tốc độ: thay phần này bằng Aho–Corasick mà không đổi API.
# ======================

def extract_by_aliases(q: str, aliases_map: Dict[str, List[str]]) -> List[str]:
    """
    Match alias theo cụm dài trước (longest-first) và tránh match chồng lấp.
    aliases_map: {canonical: [alias1, alias2, ...]}
    return: [canonical...] dedup giữ thứ tự (theo thứ tự match được)
    """
    qn = _norm(q)

    # 1) Flatten + normalize alias
    items: List[Tuple[str, str]] = []  # (alias_norm, canonical)
    for canonical, aliases in aliases_map.items():
        for a in aliases:
            a_n = _norm(a)
            if a_n:
                items.append((a_n, canonical))

    # 2) Sort alias theo độ dài giảm dần để match cụm dài trước
    #    (tie-break: alias text để ổn định)
    items.sort(key=lambda x: (-len(x[0]), x[0]))

    taken_spans: List[Tuple[int, int]] = []   # các đoạn đã “chiếm” trong qn
    picked: List[str] = []
    picked_set = set()

    def overlaps(s: int, e: int) -> bool:
        for s2, e2 in taken_spans:
            if not (e <= s2 or s >= e2):  # có giao nhau
                return True
        return False

    for alias_n, canonical in items:
        # \b... \b để match theo cụm từ (sau khi normalize còn ASCII + space)
        pat = re.compile(rf"\b{re.escape(alias_n)}\b")

        for m in pat.finditer(qn):
            s, e = m.start(), m.end()
            if overlaps(s, e):
                continue

            # Nhận match này
            taken_spans.append((s, e))

            # Chỉ add canonical 1 lần, giữ thứ tự theo lần đầu match được
            if canonical not in picked_set:
                picked.append(canonical)
                picked_set.add(canonical)

            # Không break ở đây nếu bạn muốn 1 alias “chiếm” nhiều đoạn,
            # nhưng thường 1 lần là đủ để chặn các alias ngắn hơn.
            break

    return picked

def extract_all_groups(q: str, aliases_by_group: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """
    aliases_by_group: {group_name: aliases_map}
    return: {group_name: [canonical...]} (dedup giữ thứ tự)
    """
    result: Dict[str, List[str]] = {}
    for group, amap in aliases_by_group.items():
        vals = extract_by_aliases(q, amap)
        if vals:
            result[group] = vals
    return result


ALIASES_BY_GROUP = {
    "chemical": CHEMICAL_ALIASES,
    "crop": CROP_ALIASES,
    "disease": DISEASE_ALIASES,
    "pest": PEST_ALIASES,
    "product": PRODUCT_ALIASES,
    "formula": FORMULA_ALIASES,   # NEW
    "brand": BRAND_ALIASES,   # NEW
    "mechanisms": MECHANISMS_ALIASES,
}

GROUP_TAG_PREFIX = {
    "alias": "alias",
    "chemical": "chemical",
    "crop": "crop",
    "disease": "disease",
    "pest": "pest",
    "product_group": "product-group",
    "product": "product",
    "formula": "formula",         # NEW
    "brand": "brand",
    "mechanisms": "mechanisms",         # NEW
}

# ======================
# 4) RULE-DRIVEN TAGGING (đây là phần bạn muốn "không lặp")
# ======================

def _dedup(lst: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for x in lst:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def apply_group_rules(q0: str, found: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """
    Input:
      q0    : normalized query
      found : {group: [canonical...]}
    Output:
      must, anyt (chưa có entity_type; bạn gắn thêm ở bước sau)
    """

    must: List[str] = []
    anyt: List[str] = []

    # ---- Các “intent signal” dùng lại cho nhiều nhóm
    has_product_term = bool(re.search(r"\b(san pham|thuoc)\b", q0))
    has_control_term = bool(re.search(r"\b(tri|diet|phong|dac tri|xu ly|tru)\b", q0))
    has_contains_term = bool(re.search(r"\b(chua|co trong|thanh phan|hoat chat)\b", q0))

    # ---- RULE CONFIG theo nhóm
    # action: "must" => đưa tag vào must, "any" => đưa tag vào anyt
    RULES = {
        # CHEMICAL: nếu câu hỏi kiểu “sản phẩm chứa hoạt chất” => entity product + chemical vào MUST
        "chemical": {
            "if": lambda: (has_product_term and has_contains_term) or
                          bool(re.search(r"\b(san pham)\b.*\b(chua|co)\b", q0)),
            "on_true": "must",
            "on_false": "any",
            "force_entity_product_on_true": True,
        },

        # PEST: nếu mang ý trị/diệt/phòng... => entity product + pest vào MUST
        "pest": {
            "if": lambda: has_control_term or
                          bool(re.search(r"\b(thuoc|san pham)\b.*\b(tri|diet|phong)\b", q0)) or
                          bool(re.search(r"\b(sau|ray|nhen|bo|rep|kien)\b", q0)),
            "on_true": "must",
            "on_false": "any",
            "force_entity_product_on_true": True,
        },

        # CROP/DISEASE/PRODUCT/PRODUCT_GROUP/ALIAS: mặc định soft (ANY)
        "crop": {"default": "any"},
        "disease": {"default": "any"},
        "product": {"default": "any"},
        "product_group": {"default": "any"},
        "alias": {"default": "any"},
        "formula": {"default": "any"},
    }

    for group, values in found.items():
        cfg = RULES.get(group, {"default": "any"})
        prefix = GROUP_TAG_PREFIX.get(group, group)

        if "if" in cfg:
            cond = bool(cfg["if"]())
            target = cfg["on_true"] if cond else cfg["on_false"]

            if cond and cfg.get("force_entity_product_on_true"):
                must.append("entity:product")

            for v in values:
                tag = f"{prefix}:{v}"
                (must if target == "must" else anyt).append(tag)

        else:
            target = cfg.get("default", "any")
            for v in values:
                tag = f"{prefix}:{v}"
                (must if target == "must" else anyt).append(tag)

    return _dedup(must), _dedup(anyt)


# ======================
# 5) ENTITY TYPE INFERENCE (giữ patterns xương sống)
#    (Bạn có thể dán infer_entity_type cũ của bạn vào đây)
# ======================

ENTITY_TYPES = ["registry", "product", "disease", "procedure", "pest", "weed", "general"]

def infer_entity_type(q: str):
    qn = _norm(q)

    patterns = {
        "procedure": [
            (r"\b(quy trinh|cac buoc|huong dan|lam the nao|cach)\b", 2),
            (r"\b(pha|phun|xit|tuoi|bon|rai|tron|xu ly|ngam)\b", 1),
            (r"\b(lieu|lieu luong|nong do|dinh ky|thoi diem)\b", 1),
            (r"\b(binh\s*(16|25)l|ml|lit|l|g|kg|ha|%)\b", 1),
        ],
        "product": [
            (r"\b(thuoc|thuoc gi|ten thuoc|san pham|hang|nha san xuat)\b", 2),
            (r"\b(hoat chat|ai|thanh phan)\b", 2),
            (r"\b(wp|wg|sc|ec|sl|gr|od|df|sp|cs|fs)\b", 1),
            (r"\b(gia|mua o dau|dai ly|phan phoi|tuong duong|thay the)\b", 1),
        ],
        "disease": [
            (r"\b(benh|nam benh|trieu chung|phong tri benh)\b", 2),
            (r"\b(thoi|dom|chay la|vang la|heo|xi mu|moc|ri sat)\b", 1),
            (r"\b(phytophthora|fusarium|anthracnose)\b", 2),
        ],
        "pest": [
            (r"\b(sau|bo|ray|ruoi|rep|bo tri|nhen|mot|sung|tuyen trung)\b", 2),
            (r"\b(phong tru sau|diet ray|tru sau)\b", 2),
        ],
        "weed": [
            (r"\b(co dai|co)\b", 2),
            (r"\b(diet co|tru co)\b", 2),
            (r"\b(tien nay mam|hau nay mam|la rong|la hep)\b", 1),
        ],
        "registry": [
            (r"\b(dang ky|giay phep|so dang ky|ma so)\b", 2),
            (r"\b(danh muc|duoc phep|cam|han che)\b", 2),
            (r"\b(thong tu|nghi dinh|co quan|cuc|gia han|thoi han)\b", 1),
            (r"\b(tra cuu|verify|hop phap|tem nhan)\b", 1),
        ],
    }

    score = {k: 0 for k in ENTITY_TYPES}
    for et, rules in patterns.items():
        for pat, w in rules:
            if re.search(pat, qn):
                score[et] += w

    ranked = sorted([(et, sc) for et, sc in score.items() if et != "general"], key=lambda x: x[1], reverse=True)
    top1, s1 = ranked[0]
    top2, s2 = ranked[1]

    if s1 <= 0:
        return "general", score
    if (s1 - s2) <= 1 and s2 > 0:
        return (top1, top2), score
    return top1, score


# ======================
# 6) MAIN API: infer_filters_from_query (pipeline gọi)
# ======================

def finalize_filters(must: List[str], anyt: List[str]):
    # 1) bỏ toàn bộ entity:*
    must = [t for t in must if not t.startswith("entity:")]
    anyt = [t for t in anyt if not t.startswith("entity:")]

    # 2) khử trùng lặp, giữ thứ tự
    def dedup(xs):
        seen=set(); out=[]
        for x in xs:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    return dedup(must), dedup(anyt)

def strip_entity_tags(tags):
    return [t for t in tags if not str(t).startswith("entity:")]

def enforce_backbone_without_entity(must, anyt):
    """
    - Không dùng entity:*
    - Nếu must rỗng, kéo 1 tag cụ thể từ anyt lên must theo thứ tự ưu tiên.
    """
    must = strip_entity_tags(must)
    anyt = strip_entity_tags(anyt)

    # Nếu must đã có tag cụ thể thì OK
    if must:
        return must, anyt

    # Thứ tự ưu tiên "xương sống"
    priority_prefixes = (
        "mechanisms",
        "formula:",
        "brand:",
        "pest:",
        "disease:",
        "weed:",
        "product:",
        "chemical:",
        "procedure:",
        "registry:",   # nếu anh có nhóm này
        # "crop:"       # thường KHÔNG nên làm backbone (rộng quá), nhưng có thể thêm nếu muốn
    )

    for prefix in priority_prefixes:
        hit_idx = next((i for i, t in enumerate(anyt) if str(t).startswith(prefix)), None)
        if hit_idx is not None:
            hit = anyt.pop(hit_idx)
            must.append(hit)
            break

    return must, anyt

def relax_must_same_group(must, anyt, prefixes=("pest:", "disease:", "weed:", "chemical:", "product:")):
    """
    Nếu MUST có nhiều tag cùng group (vd: pest:*),
    giữ 1 cái trong MUST, các cái còn lại chuyển sang ANY.
    """
    new_must = []
    moved = []

    for p in prefixes:
        same = [t for t in must if t.startswith(p)]
        if len(same) > 1:
            keep = same[0]          # giữ cái đầu (hoặc chọn theo score nếu có)
            new_must.append(keep)
            moved.extend(same[1:])  # phần dư đẩy sang ANY
        elif len(same) == 1:
            new_must.append(same[0])

    # giữ lại các must không thuộc group trên
    others = [t for t in must if not any(t.startswith(p) for p in prefixes)]
    new_must.extend(others)

    # cập nhật anyt
    anyt = anyt + moved
    anyt = list(dict.fromkeys(anyt))  # dedup, giữ thứ tự

    return new_must, anyt

def reorder_any_by_priority(anyt: List[str], priority_prefixes=("mechanisms:", "formula:")) -> List[str]:
    """
    Đưa các tag có prefix ưu tiên lên đầu (giữ thứ tự tương đối).
    Mặc định: mechanisms:* trước, rồi formula:*, rồi các tag khác.
    """
    hi = []
    lo = []
    for t in anyt:
        if any(str(t).startswith(p) for p in priority_prefixes):
            hi.append(t)
        else:
            lo.append(t)
    return hi + lo

def infer_answer_intent(q: str, found_groups: Dict[str, List[str]]):
    qn = _norm(q)

    has_product_term = bool(re.search(r"\b(thuoc|san pham)\b", qn))
    has_control_term = bool(re.search(r"\b(tri|diet|phong|xu ly|tru)\b", qn))
    has_symptom_term = bool(re.search(r"\b(trieu chung|benh)\b", qn))

    if has_product_term and has_control_term:
        return "product"

    if has_symptom_term:
        return "disease"

    return "general"


def infer_filters_from_query(q: str):
    q0 = _norm(q)

    found = extract_all_groups(q0, ALIASES_BY_GROUP)

    must, anyt = apply_group_rules(q0, found)
    must, anyt = finalize_filters(must, anyt)

    # entity chỉ để log/debug
    et, _score = infer_entity_type(q0)
    debug_log(
        f"ENTITY TYPE : {et}",
        f"ENTITY SCORE: {_score}"
    )

    def strip_entity(tags):
        return [t for t in tags if not str(t).startswith("entity:")]

    must = strip_entity(must)
    anyt = strip_entity(anyt)

    # GIỮ SEMANTICS BẢN CŨ:
    # - Không dùng MUST ở downstream (tránh over-filter). Gom toàn bộ MUST sang ANY.
    # - Sau đó chỉ ưu tiên thứ tự mechanisms/brand trong ANY để tăng khả năng khớp.
    anyt = must + anyt
    must = []

    # Ưu tiên mechanisms trước mọi thứ trong ANY (và brand nếu có)
    anyt = reorder_any_by_priority(anyt, priority_prefixes=("mechanisms:", "formula:"))

    # dedupe giữ thứ tự
    seen = set()
    anyt2 = []
    for t in anyt:
        if t not in seen:
            anyt2.append(t)
            seen.add(t)

    debug_log(
        f"FINAL MUST : {must}",
        f"FINAL ANY  : {anyt2}"
    )
    return {
        "must": must,
        "any": anyt,
        "found": found,          # 👈 THÊM
        "entity_type": et,       # (tuỳ chọn)
        "entity_score": _score,  # (tuỳ chọn)
    }


