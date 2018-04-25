import json

filePath='name_list'
with open(filePath) as f:
    dic={'AMBIENCE#GENERAL':[['atmosphere','romantic','ambience'],['restaurant','place','food','service','decor','service']],
         'DRINKS#PRICES':[['wine','wines','water','drinks'],['prices']],
         'DRINKS#QUALITY':[['drinks','wine','sake'],['service','glass','bar']],
         'DRINKS#STYLE_OPTIONS':[['wine','wines','bear','drinks'],['list','menu','selection','bar']],
         'FOOD#PRICES':[['food'],['price','overprice','prices','worth']],
         'FOOD#QUALITY':[['food','pizza','sushi','dishes','chicken'],['amazing','fresh','excellent','delicious','good','great','best']],
         'FOOD#STYLE_OPTIONS':[['food','menu','portions','rolls'],['small','even','fresh','large','limited','good','great','huge','delicious','fresh']],
         'LOCATION#GENERAL':[['location','place','city'],['view','avenue','river','neighborhood','restaurant']],
         'RESTAURANT#GENERAL':[['restaurant','place'],['experience','time','food']],
         'RESTAURANT#MISCELLANEOUS':[['restaurant'],['place','spot','friends','time','dinner','food','occasion']],
         'RESTAURANT#PRICES':[['restaurant','place',],['prices','price','worth','money','value']],
         'SERVICE#GENERAL':[['service'],['food','staff','us','place','waiter','restaurant','table']]
         }

    json.dump(dic,filePath,indent=4)