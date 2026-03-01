import os
import json
import openai
import time
from typing import List
from pathlib import Path
import argparse

# 设定默认路径和API key
OUTPUT_DIR_DEFAULT = Path("/Users/50pai/Desktop/Writing book agent/local_llm_xian_ni/data/create/novel_outlines")
API_KEY = "YOUR_DEEPSEEK_API_KEY"  # 在这里设置你的API Key

# 读取小说总大纲和知识库
def read_text_file(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="utf-8-sig")

# 生成章节大纲的请求
def generate_chapter_outlines(outline_text: str, kb_text: str, num_chapters: int = 10) -> List[str]:
    prompt = f"""
你是一名专业的长篇网络小说策划师。

我将给你一部小说的【整本总大纲】以及【人物与世界观知识库】。
你的任务是：只生成“章节级写作大纲”，不要写正文。

要求：
1. 请从第1章开始，连续生成到第{num_chapters}章
2. 每一章都必须单独成块，格式严格如下：

【第0001章 写作大纲】
- 本章目标：
- 主要冲突：
- 推进事件链（按顺序）：
- 情绪与节奏：
- 结尾钩子：

3. 章节之间必须强连贯，后一章必须承接前一章的结尾钩子
4. 不要写正文，不要出现对话
5. 不要省略章节
6. 只输出章节大纲内容，不要额外解释

【整本小说总大纲】
{outline_text}

【人物 / 世界观 / 规则知识库】
{kb_text}

现在开始生成第1章到第{num_chapters}章的章节大纲。
"""

    # 请求OpenAI API
    response = openai.Completion.create(
        model="text-davinci-003",  # 更换为你需要的模型
        prompt=prompt,
        max_tokens=2000,  # 可根据需要调整
        api_key=API_KEY
    )
    
    return response.choices[0].text.strip().split("\n\n")

# 保存生成的章节大纲到文件
def save_chapter_outlines(outlines: List[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, outline in enumerate(outlines):
        with open(output_dir / f"chapter_outline_{i+1:04d}.txt", "w", encoding="utf-8") as f:
            f.write(outline)

# 主函数
def main():
    parser = argparse.ArgumentParser(description="Generate chapter outlines based on the novel outline and knowledge base.")
    parser.add_argument("--outline_file", required=True, help="Path to the novel outline file")
    parser.add_argument("--kb_dir", required=True, help="Directory containing the knowledge base files (characters, world-building, etc.)")
    parser.add_argument("--out_dir", default=str(OUTPUT_DIR_DEFAULT), help="Directory to save generated chapter outlines")
    parser.add_argument("--chapters", type=int, default=10, help="Number of chapters to generate (default is 10)")
    
    args = parser.parse_args()

    # 读取总大纲和知识库
    outline_text = read_text_file(Path(args.outline_file))
    kb_text = "\n".join([read_text_file(Path(args.kb_dir) / f) for f in os.listdir(args.kb_dir)])

    # 生成章节大纲
    outlines = generate_chapter_outlines(outline_text, kb_text, num_chapters=args.chapters)
    
    # 保存生成的大纲
    save_chapter_outlines(outlines, Path(args.out_dir))

    print(f"章节大纲已生成并保存到: {args.out_dir}")

if __name__ == "__main__":
    main()
