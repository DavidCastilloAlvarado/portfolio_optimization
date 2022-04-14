from bs4 import BeautifulSoup as bs
from urllib.request import urlopen, Request
import numpy as np
import time


def request_url(url):
    # print(url)
    if not url.startswith("http"):
        raise RuntimeError(
            "Incorrect and possibly insecure protocol in url " + url)
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Authority": "www.tipranks.com",
        "user-agent": "ozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36",
        "sec-fetch-dest": "document",
        "sec-fetch-user": "?1",
        'sec-ch-ua-platform': "Linux",
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="96", "Google Chrome";v="96"',
        'cookie': 'GCLB=CPzZjeaDjpTC9AE; TiPMix=93.9823603229515; x-ms-routing-name=self; _gid=GA1.2.1087446361.1638810368; _fbp=fb.1.1638810367631.1305239362; tr-experiments-version=1.13; tipranks-experiments=%7b%22Experiments%22%3a%5b%7b%22Name%22%3a%22first-few-analyst-ratings%22%2c%22Variant%22%3a%22default%22%2c%22SendAnalytics%22%3afalse%7d%2c%7b%22Name%22%3a%22go-pro-variant%22%2c%22Variant%22%3a%22v2%22%2c%22SendAnalytics%22%3afalse%7d%2c%7b%22Name%22%3a%22checkout-page-variant%22%2c%22Variant%22%3a%22tipranks%22%2c%22SendAnalytics%22%3atrue%7d%5d%7d; prism_90278194=edb57f14-8001-4cb3-b39f-fadeec8a0898; _hjFirstSeen=1; _hjSession_2550200=eyJpZCI6ImQ0ZGU5ZTAwLWU2NDEtNDZhZC1iYWY3LTZlOTVkZGY0ZThjNSIsImNyZWF0ZWQiOjE2Mzg4MTAzNjg2OTR9; _hjIncludedInSessionSample=0; _hjAbsoluteSessionInProgress=0; rbzid=YViC4mOIeKVvQrMH8cl83/3N0I0zbqQYb+DKTWrdIinhOCpmVAG1HJaEU4jeh9RBE95wfs0BDC/u/jGkl22x/HAzyJMIPIx3znuk+JZ0lxN5MStCkKiSezBz7nFFSSL7xkT+ktyoGYfvyJD1kOMMa42ZgQ7bSfb5Jd16VNscRYy/fGY55aJ5p/4GOj16enHpjqBm+EqEdKKCuyJTXoq2/zkHUCX0SM4spZoCyc8JlTQBLobmLOjc+JIofN7maMC3KChCwtDEeq1O5mEZVmc2bA==; rbzsessionid=a5a372f9cb37556462d7092e251a5957; abtests=1,1; filters={%22sector%22:%22general%22%2C%22period%22:%22yearly%22%2C%22benchmark%22:%22none%22}; _hjSessionUser_2550200=eyJpZCI6ImNjMWNlZjFiLWZlOWQtNWU4YS04NGNkLWVhODc3M2I3MmJkZiIsImNyZWF0ZWQiOjE2Mzg4MTAzNjg2NTYsImV4aXN0aW5nIjp0cnVlfQ==; _ce.s=v11.rlc~1638810374720; ai_user=GeH2M|2021-12-06T17:06:15.082Z; _hjCachedUserAttributes=eyJhdHRyaWJ1dGVzIjp7ImNsaWNrZWRPbkRvd25sb2FkQXBwIjoiRmFsc2UiLCJuZXdTaWRlYmFyRGVza3RvcEZvcmVjYXN0Ijp0cnVlLCJ1c2VyX3BsYW4iOiJvcGVuIn0sInVzZXJJZCI6ImZ1bmN0aW9uKGEpe2Euc2V0KFwiZGltZW5zaW9uXCIrYixhLmdldChcImNsaWVudElkXCIpKX0ifQ==; _hjDonePolls=755041; _gat_UA-38500593-6=1; _gat_UA-38500593-18=1; _ga_FFX3CZN1WY=GS1.1.1638810367.1.1.1638810817.0; _ga=GA1.2.2031307493.1638810368; __gads=ID=8c5d86524d8797b7-224821356c7b00af:T=1638810433:RT=1638810818:S=ALNI_Mb0Njzz4vzJj1eIp72qm_7VrFzRwg',
        "accept-language": "en-US,en;q=0.9,es;q=0.8",
        "cache-control": "max-age=0",
        "sec-ch-ua": "\" Not A;Brand\";v=\"99\", \"Chromium\";v=\"96\", \"Google Chrome\";v=\"96\"",
        "sec-ch-ua-mobile": "?0",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "upgrade-insecure-requests": "1"
    }
    httprequest = Request(url, headers=headers)

    with urlopen(httprequest) as response:
        if response.status == 200:
            val = response.read().decode()
            time.sleep(1)
            return val
        else:
            raise RuntimeError("Error in request " + url)


def forecast_12months(stock):
    url = f"https://www.tipranks.com/stocks/{stock.lower()}/forecast"
    # print(url)
    html = request_url(url)

    soup = bs(html, 'lxml')
    secret = soup.select_one('#tr-stock-page-content > div.maxW1200.grow1.flexc__.flexc__.displayflex > div.minW80.z1.flexr__f.maxW1200.mobile_maxWparent > div._1HABim5jz41fqgmk1lBtqo.tr-box-ui.flexc__.w6.px0.displayflex.minHmedium.z0.mb7.pl4.ipad_pl2.ipad_w_pxscreen.ipad_minHauto.grow1.mobile_pr0.mobile_pl0.mobile_w12 > div.flexc__.mt3.bgwhite.displayflex.border1.borderColorwhite-8.shadow1.positionrelative.grow1 > div.w12.p0.displayflex.positionrelative.grow1 > div > div > div.w12.displayflex.ipad_w3.mobile_w12.mobile_flexcsc > div.displayinline-block.flexrcc.colorblack-5.fonth8_normal.ipad_lineHeight4.ipad_fontSize8.ml4.mt2.bl1_solid.pl4.borderColorgray-0.ipad_w8.mobile_order4.mobile_w12.mobile_pl0.mobile_pr3.mobile_bordernone.mobile_w_pxauto.mobile_mb3.mobile_mt4 > span.fontWeightsemibold.colorgray-1')
    forecast_val = secret.text
    forecast_val = forecast_val[1:].replace(",", "")
    return float(forecast_val)


def MarketCapExtract(html):

    soup = bs(html, 'lxml')
    secret = soup.select_one('#quote-summary > div.D\(ib\).W\(1\/2\).Bxz\(bb\).Pstart\(12px\).Va\(t\).ie-7_D\(i\).ie-7_Pos\(a\).smartphone_D\(b\).smartphone_W\(100\%\).smartphone_Pstart\(0px\).smartphone_BdB.smartphone_Bdc\(\$seperatorColor\) > table > tbody > tr:nth-child(1) > td.Ta\(end\).Fw\(600\).Lh\(14px\)')
    marketcap = secret.text
    size = marketcap[-1]
    if size == "B":
        return float(marketcap[:-1]) * 10**9
    elif size == "M":
        return float(marketcap[:-1]) * 10**6
    elif size == "T":
        return float(marketcap[:-1]) * 10**12


def create_views(shares, changes):
    views = []
    for i, share in enumerate(shares):
        for i_p, share_p in enumerate(shares[i:], i):
            if share == share_p:
                continue
            if changes[i] > changes[i_p]:
                compare = '>'
            else:
                compare = '<'
            view = (share, compare, share_p, abs(changes[i] - changes[i_p]))
            views.append(view)
    return views


def create_views_and_link_matrix(shares, views):
    r, c = len(views), len(shares)
    Q = [views[i][3] for i in range(r)]  # view matrix
    P = np.zeros([r, c])
    nameToIndex = dict()
    for i, n in enumerate(shares):
        nameToIndex[n] = i
    for i, v in enumerate(views):
        name1, name2 = views[i][0], views[i][2]
        P[i, nameToIndex[name1]] = +1 if views[i][1] == '>' else -1
        P[i, nameToIndex[name2]] = -1 if views[i][1] == '>' else +1
    return np.array(Q), P
