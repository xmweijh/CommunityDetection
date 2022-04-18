import urllib.request
import re
import os
import xlwt


def get_pdf_arxiv(web_site, path):
    rep = urllib.request.urlopen(urllib.request.Request(web_site))
    page = rep.read().decode('utf-8')
    pdf_download = re.findall('<meta name="citation_pdf_url" content="(.*?)"/>', page, re.S)  # 查询到网页中对应的pdf下载链接
    print(pdf_download[0])
    if (len(pdf_download) != 0):
        try:
            u = urllib.request.urlopen(pdf_download[0])
        except urllib.error.HTTPError:
            print(pdf_download[0], "url file not found")
            return
        block_sz = 8192
        with open(path, 'wb') as f:
            while True:
                buffer = u.read(block_sz)
                if buffer:
                    f.write(buffer)
                else:
                    break
        print("Sucessful to download " + path)


# 创建一个Excel文件
file = xlwt.Workbook()
# 创建sheet工作表
sheet1 = file.add_sheet(u'表1', cell_overwrite_ok=True)

for j in range(10):
    req = urllib.request.Request(
        'https://dblp.uni-trier.de/search//publ/inc?q=Community%20Detection&s=ydvspc&h=30&b=' + str(
            j))  # 此处只需要修改q=后面的内容，并且保留&s=ydvspc之后的内容
    response = urllib.request.urlopen(req)
    the_page = response.read().decode('utf-8')

    paper_title = re.findall('<span class="title" itemprop="name">(.*?)</span>', the_page, re.S)  # 检索页面中给的论文名字
    paper_pub = re.findall('<span itemprop="name">(.*?)</span>', the_page, re.S)  # 检索出版商
    paper_year = re.findall('<span itemprop="datePublished">(.*?)</span>', the_page, re.S)  # 检索出版年份
    # paper_web = re.findall('view</b></p><ul><li class="ee"><a href="(.*?)" itemprop="url">', the_page, re.S)  # 存储文章对应网址

    for i in range(len(paper_title)):
        # 以下代码去除了原有网页论文名字中无法保存为window文件名的符号，采用下划线_代替
        paper_title[i] = paper_title[i].replace('"', '')
        list = paper_title[i].split(" ")
        paper_title[i] = "_".join(list)
        list = paper_title[i].split(":")
        paper_title[i] = "_".join(list)
        list = paper_title[i].split("?")
        paper_title[i] = "_".join(list)
        list = paper_title[i].split("<sub>")
        paper_title[i] = "_".join(list)
        list = paper_title[i].split("</sub>")
        paper_title[i] = "_".join(list)
        list = paper_title[i].split("<sup>")
        paper_title[i] = "_".join(list)
        list = paper_title[i].split("</sup>")
        paper_title[i] = "_".join(list)
        print(paper_title[i])
        # path_dir = "D:\\APTX-4869\\md\\机器学习\\社区网络\\论文\\"

        # 写入数据从 0 开始计数
        sheet1.write(i + j*30, 0, paper_title[i])

        if i < len(paper_pub):
            sheet1.write(i + j*30, 1, paper_pub[i])
        if i < len(paper_year):
            sheet1.write(i + j*30, 2, paper_year[i])

        # dir_list = os.listdir(path_dir)
        # path = paper_title[i] + "pdf"
        # if path not in dir_list:
        #     get_pdf_arxiv(paper_web[i], path_dir + path)
    # 保存文件
file.save(f"论文.xls")
