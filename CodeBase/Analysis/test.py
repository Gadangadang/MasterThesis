import pathlib 
import requests



#message = "Hello, this is MadGanga"
#url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
#print(requests.get(url).json()) # this sends the message

path = STORE_IMG_PATH/"histo/b_data_recon_big_rm3_feats_sig_Dummydata.pdf"

files = {"photo":open(path, "rb")}

resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chat_id}&caption=Dummy Data", files=files)
print(resp.status_code)