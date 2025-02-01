import nltk
from nltk import pos_tag, word_tokenize
from collections import defaultdict, Counter
import json

response_path = '/home/dyf/data_generate/doc-instruct/data/lima/epoch/com/com_ablation/new_response/diverse_filter/1w_com2_diverse.jsonl'
data = [json.loads(l) for l in open(response_path, "r")]

# 假设数据是一个包含文本的列表
texts = data  # 如果数据是字典或其他结构，请根据实际情况提取文本
excluded_verbs = ['be', 'do', 'does', 'consider', 'considers', 'considered', 'considering', 'is', 'are', 'am']

# 提取动词和名词
verb_noun_counts = defaultdict(Counter)

for text in texts:
    tokens = word_tokenize(text['conversations'][0])
    tagged = pos_tag(tokens)
    for word, tag in tagged:
        if tag.startswith('VB'):  # 动词
            verb = word.lower()
            if verb in excluded_verbs:  # 如果动词在排除列表中，跳过
                continue
        elif tag.startswith('NN'):  # 名词
            noun = word.lower()
            if 'verb' in locals():  # 确保动词已经出现且未被排除
                verb_noun_counts[verb][noun] += 1
                del verb  # 重置动词

# 统计最常见的20个动词
all_verbs = [verb for verb in verb_noun_counts.keys()]
top_verbs = Counter(all_verbs).most_common(25)
top_verbs = [verb for verb, count in top_verbs]

# 获取每个动词的top 5名词
top_nouns_per_verb = {verb: verb_noun_counts[verb].most_common(5) for verb in top_verbs}

from pyecharts import options as opts
from pyecharts.charts import Sunburst

# 准备数据
data = []
for verb in top_verbs:
    verb_data = {
        "name": verb,
        "children": [{"name": noun, "value": count} for noun, count in top_nouns_per_verb[verb]]
    }
    data.append(verb_data)
    
noun_counts = [count for verb in top_verbs for _, count in top_nouns_per_verb[verb]]
max_verb_count = max(verb_total_counts.values()) if verb_total_counts else 0
max_noun_count = max(noun_counts) if noun_counts else 0
max_count = max(max_verb_count, max_noun_count)
# 动态计算字体大小的函数
def dynamic_font_size(value):
    min_size = 10
    max_size = 20
    if max_count == 0:
        return min_size
    return int(min_size + (max_size - min_size) * (value / max_count))

chart = (
    Sunburst(init_opts=opts.InitOpts(width="1500px", height="1000px"))
    .add(
        series_name="词频分布",
        data_pair=data,
        radius=[0, "90%"],
        label_opts=opts.LabelOpts(
            formatter="{b}\n({c})",
            font_size=dynamic_font_size,
            position="inside",
            rotate="radial"
        )
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="动词-名词词频分布图"),
        tooltip_opts=opts.TooltipOpts(formatter="{b}: {c}")
    )
)

# 渲染图表
chart.render("/home/dyf/data_generate/doc-instruct/ablation_test/sunburst_chart.html")