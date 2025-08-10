import os
import re
import glob
import subprocess
from pathlib import Path

# --- API配置 ---
try:
    from openai import OpenAI
    QWEN_API_KEY = os.getenv('DASHSCOPE_API_KEY', "sk-341bf65fcfd4478e9c06058da2c91a24")
    client = OpenAI(
        api_key=QWEN_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    print("未安装openai库，将使用基础转换方式")
except Exception as e:
    API_AVAILABLE = False
    print(f"API初始化失败: {e}，将使用基础转换方式")

# 检查是否有可用的PDF处理工具
def check_pdf_tools():
    """检查系统中可用的PDF处理工具"""
    tools = []
    
    # 检查MinerU (magic-pdf)
    try:
        result = subprocess.run(['magic-pdf', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            tools.append('MinerU')
    except:
        pass
    
    # 检查PyPDF2
    try:
        import PyPDF2
        tools.append('PyPDF2')
    except ImportError:
        pass
    
    # 检查pdfplumber
    try:
        import pdfplumber
        tools.append('pdfplumber')
    except ImportError:
        pass
    
    # 检查系统命令
    try:
        result = subprocess.run(['pdftotext', '-v'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            tools.append('pdftotext')
    except:
        pass
    
    return tools

def install_instructions():
    """显示安装说明"""
    print("请安装以下工具之一来处理PDF文件：")
    print("1. MinerU (推荐，高质量转换): pip install -U 'magic-pdf[full]'")
    print("   - 需要下载模型权重，详见: https://github.com/opendatalab/MinerU")
    print("2. PyPDF2: pip install PyPDF2")
    print("3. pdfplumber: pip install pdfplumber")
    print("4. poppler-utils (包含pdftotext):")
    print("   - Windows: 下载并安装poppler")
    print("   - Linux: sudo apt-get install poppler-utils")
    print("   - Mac: brew install poppler")

def pdf_to_markdown(pdf_path, md_path):
    """将PDF文件转换为Markdown文件"""
    print(f"正在转换 {pdf_path} 到 {md_path}")
    
    tools = check_pdf_tools()
    if not tools:
        install_instructions()
        return False
    
    markdown_content = ""
    
    try:
        # 使用第一个可用的工具
        tool = tools[0]
        print(f"使用工具: {tool}")
        
        if tool == 'MinerU':
            # 使用MinerU进行高质量转换
            try:
                result = subprocess.run(['magic-pdf', pdf_path, '-o', str(md_path)], capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    print(f"MinerU转换成功: {md_path}")
                    return True
                else:
                    print(f"MinerU转换失败: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                print("MinerU转换超时，请重试")
                return False
            except Exception as e:
                print(f"MinerU转换出错: {e}")
                return False
        
        elif tool == 'PyPDF2':
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for i, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        markdown_content += process_text_to_markdown(text)
        
        elif tool == 'pdfplumber':
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        markdown_content += process_text_to_markdown(text)
        
        elif tool == 'pdftotext':
            # 使用系统命令pdftotext
            temp_txt = md_path.replace('.md', '_temp.txt')
            result = subprocess.run(['pdftotext', pdf_path, temp_txt], capture_output=True, text=True)
            if result.returncode == 0:
                with open(temp_txt, 'r', encoding='utf-8') as f:
                    text = f.read()
                markdown_content = process_text_to_markdown(text)
                os.remove(temp_txt)
            else:
                print(f"pdftotext执行失败: {result.stderr}")
                return False
    
    except Exception as e:
        print(f"处理PDF文件时出错: {e}")
        return False
    
    # 写入markdown文件
    try:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"成功生成Markdown文件: {md_path}")
        return True
    except Exception as e:
        print(f"写入Markdown文件时出错: {e}")
        return False

def process_text_to_markdown(text):
    """将文本处理为Markdown格式"""
    if API_AVAILABLE and len(text.strip()) > 100:
        return enhance_with_api(text)
    else:
        return basic_text_to_markdown(text)

def basic_text_to_markdown(text):
    """基础的文本到Markdown转换"""
    markdown_content = ""
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 简单的标题检测逻辑
        if (line.isupper() and len(line) < 100) or \
           (len(line) < 50 and not line.endswith('.')):
            markdown_content += f"## {line}\n\n"
        else:
            markdown_content += f"{line}\n\n"
    
    markdown_content += "\n---\n\n"  # 页面分隔符
    return markdown_content

def enhance_with_api(text):
    """使用大模型API增强文本到Markdown的转换"""
    try:
        print("使用大模型API增强转换质量...")
        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": "你是一个专业的文档格式转换助手。请将以下PDF提取的文本转换为结构化的Markdown格式，包括适当的标题、段落、列表等。保持原文内容不变，只改善格式结构。"},
                {"role": "user", "content": f"请将以下文本转换为Markdown格式：\n\n{text}"}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        enhanced_content = response.choices[0].message.content
        return enhanced_content + "\n\n---\n\n"  # 添加页面分隔符
    except Exception as e:
        print(f"API增强失败，使用基础转换: {e}")
        return basic_text_to_markdown(text)

def markdown_to_latex(md_path, latex_path):
    """将Markdown文件转换为LaTeX文件"""
    print(f"正在转换 {md_path} 到 {latex_path}")
    
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
    except Exception as e:
        print(f"读取Markdown文件时出错: {e}")
        return False
    
    # LaTeX文档模板（增强版，支持数学公式）
    latex_template = r"""\documentclass{article}
\usepackage{ctex}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathtools}

\title{PDF转换文档}
\author{自动转换}
\date{\today}

\begin{document}

\maketitle

{content}

\end{document}"""
    
    # 转换markdown到latex
    latex_content = md_content
    
    # 转换各级标题 - 修复Markdown标题格式，支持空格
    latex_content = re.sub(r'^\s*#\s+(.+)$', r'\\section{\1}', latex_content, flags=re.MULTILINE)
    latex_content = re.sub(r'^\s*##\s+(.+)$', r'\\subsection{\1}', latex_content, flags=re.MULTILINE)
    latex_content = re.sub(r'^\s*###\s+(.+)$', r'\\subsubsection{\1}', latex_content, flags=re.MULTILINE)
    latex_content = re.sub(r'^\s*####\s+(.+)$', r'\\paragraph{\1}', latex_content, flags=re.MULTILINE)
    
    # 转换粗体和斜体
    latex_content = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', latex_content)
    latex_content = re.sub(r'\*(.+?)\*', r'\\textit{\1}', latex_content)
    
    # 转换页面分隔符
    latex_content = latex_content.replace('---', '\\newpage')
    
    # 数学公式直接保留原样，不进行转换处理
    
    # 简化的特殊字符处理，不区分数学环境
    latex_content = latex_content.replace('_', '\\_')
    latex_content = latex_content.replace('&', '\\&')
    latex_content = latex_content.replace('%', '\\%')
    latex_content = latex_content.replace('#', '\\#')
    latex_content = latex_content.replace('$', '\\$')
    latex_content = latex_content.replace('{', '\\{')
    latex_content = latex_content.replace('}', '\\}')
    latex_content = latex_content.replace('~', '\\~')
    latex_content = latex_content.replace('^', '\\^')
    
    # 清理多余的转义字符
    latex_content = re.sub(r'\\\\([{}])', r'\\\1', latex_content)
    
    # 填充模板（使用双大括号避免格式化冲突）
    final_latex = latex_template.replace('{content}', latex_content)
    
    # 写入latex文件
    try:
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(final_latex)
        print(f"成功生成LaTeX文件: {latex_path}")
        return True
    except Exception as e:
        print(f"写入LaTeX文件时出错: {e}")
        return False

def process_all_pdfs():
    """处理当前文件夹下的所有PDF文件"""
    current_dir = Path.cwd()
    pdf_files = glob.glob(str(current_dir / "*.pdf"))
    
    if not pdf_files:
        print("当前文件夹下没有找到PDF文件")
        return
    
    print(f"找到 {len(pdf_files)} 个PDF文件")
    
    for pdf_file in pdf_files:
        pdf_path = Path(pdf_file)
        base_name = pdf_path.stem
        
        # 生成对应的markdown文件路径
        md_path = current_dir / f"{base_name}.md"
        
        # PDF转Markdown
        if pdf_to_markdown(pdf_path, md_path):
            # 生成对应的latex文件路径
            latex_path = current_dir / f"{base_name}.tex"
            
            # Markdown转LaTeX
            markdown_to_latex(md_path, latex_path)
            
            print(f"完成处理: {pdf_file} -> {md_path.name} -> {latex_path.name}")
        else:
            print(f"处理 {pdf_file} 失败")
        
        print("-" * 50)

if __name__ == "__main__":
    print("PDF转Markdown转LaTeX工具")
    print("=" * 50)
    process_all_pdfs()
    print("处理完成！")