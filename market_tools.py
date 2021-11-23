from bs4 import BeautifulSoup as bs
from urllib.request import urlopen, Request
import numpy as np


def request_url(url):
    if not url.startswith("http"):
        raise RuntimeError(
            "Incorrect and possibly insecure protocol in url " + url)

    httprequest = Request(url, headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
                                        "Authority": "www.tipranks.com",
                                        "user-agent": "ozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36",
                                        "sec-fetch-dest": "document",
                                        "sec-fetch-user": "?1",
                                        'sec-ch-ua-platform': "Linux",
                                        'sec-fetch-dest': 'document',
                                        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="96", "Google Chrome";v="96"'})

    with urlopen(httprequest) as response:
        if response.status == 200:
            val = response.read().decode()
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
