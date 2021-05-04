from data_process import DataProcess


from lxml import etree
parser = etree.XMLParser(recover=True)


file = '19940513_AFP_ARB.sgm'
xtree = etree.parse('../../data_raw/ldc_arabic_newswire_a_2001_t55/transcripts/1994/{}'.format(file))
xroot = xtree.getroot()
print(xroot)
rows = []
for node in xroot.findall("HEADER"):

    l = node.text
    rows.append({"original_sentence": line})
    # print(l)
