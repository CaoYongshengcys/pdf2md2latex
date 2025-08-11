import os
import re
import json
import time
import logging
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename
import PyPDF2
import pdfplumber
import pypandoc
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here'

# 全局变量存储日志
log_entries = []
log_lock = threading.Lock()

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class LogManager:
    """日志管理器"""
    
    def __init__(self):
        self.entries = []
        self.lock = threading.Lock()
    
    def add_log(self, message: str, level: str = 'info'):
        """添加日志条目"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'message': message,
            'level': level
        }
        
        with self.lock:
            self.entries.append(log_entry)
            # 限制日志数量
            if len(self.entries) > 1000:
                self.entries = self.entries[-500:]
        
        # 同时记录到文件日志
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(message)
    
    def get_logs(self, limit: int = 100) -> List[Dict]:
        """获取最近的日志"""
        with self.lock:
            return self.entries[-limit:] if self.entries else []
    
    def clear_logs(self):
        """清空日志"""
        with self.lock:
            self.entries.clear()

log_manager = LogManager()

class PDFConverter:
    """PDF转换器"""
    
    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager
    
    def extract_text_with_pypdf2(self, pdf_path: str) -> str:
        """使用PyPDF2提取文本"""
        try:
            self.log_manager.add_log("开始使用PyPDF2提取文本", "info")
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                self.log_manager.add_log(f"PDF总页数: {num_pages}", "info")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n## 第 {page_num} 页\n\n{page_text}\n\n"
                        self.log_manager.add_log(f"已处理第 {page_num} 页", "debug")
                    except Exception as e:
                        self.log_manager.add_log(f"处理第 {page_num} 页时出错: {str(e)}", "warning")
                        continue
            
            self.log_manager.add_log("PyPDF2文本提取完成", "success")
            return text.strip()
        
        except Exception as e:
            self.log_manager.add_log(f"PyPDF2提取失败: {str(e)}", "error")
            return ""
    
    def extract_text_with_pdfplumber(self, pdf_path: str) -> str:
        """使用pdfplumber提取文本"""
        try:
            self.log_manager.add_log("开始使用pdfplumber提取文本", "info")
            text = ""
            
            with pdfplumber.open(pdf_path) as pdf:
                num_pages = len(pdf.pages)
                self.log_manager.add_log(f"PDF总页数: {num_pages}", "info")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n## 第 {page_num} 页\n\n{page_text}\n\n"
                        self.log_manager.add_log(f"已处理第 {page_num} 页", "debug")
                    except Exception as e:
                        self.log_manager.add_log(f"处理第 {page_num} 页时出错: {str(e)}", "warning")
                        continue
            
            self.log_manager.add_log("pdfplumber文本提取完成", "success")
            return text.strip()
        
        except Exception as e:
            self.log_manager.add_log(f"pdfplumber提取失败: {str(e)}", "error")
            return ""
    
    def pdf_to_markdown(self, pdf_path: str) -> str:
        """将PDF转换为Markdown"""
        self.log_manager.add_log("开始PDF到Markdown转换", "info")
        
        # 尝试使用pdfplumber（通常效果更好）
        text = self.extract_text_with_pdfplumber(pdf_path)
        
        # 如果pdfplumber失败，尝试PyPDF2
        if not text.strip():
            self.log_manager.add_log("pdfplumber提取失败，尝试使用PyPDF2", "warning")
            text = self.extract_text_with_pypdf2(pdf_path)
        
        if not text.strip():
            self.log_manager.add_log("所有PDF文本提取方法都失败了", "error")
            return ""
        
        # 清理和格式化文本
        markdown_text = self.clean_and_format_text(text)
        
        self.log_manager.add_log("PDF到Markdown转换完成", "success")
        return markdown_text
    
    def clean_and_format_text(self, text: str) -> str:
        """清理和格式化文本为Markdown"""
        self.log_manager.add_log("开始清理和格式化文本", "info")
        
        # 移除多余的空白行
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # 移除行首行尾的空白
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # 只保留非空行
                # 检测可能的标题（全大写或以数字开头的行）
                if (line.isupper() and len(line) > 3) or \
                   re.match(r'^\d+\.?\s+', line):
                    cleaned_lines.append(f"## {line}")
                else:
                    cleaned_lines.append(line)
        
        formatted_text = '\n\n'.join(cleaned_lines)
        
        # 处理可能的数学公式
        formatted_text = self.format_math_formulas(formatted_text)
        
        self.log_manager.add_log("文本清理和格式化完成", "success")
        return formatted_text
    
    def format_math_formulas(self, text: str) -> str:
        """格式化数学公式"""
        # 简单的数学公式检测和格式化
        # 这是一个基础实现，可以根据需要扩展
        
        # 检测行内数学公式（类似 $formula$ 的模式）
        text = re.sub(r'\$([^$]+)\$', r'\(\1\)', text)
        
        # 检测块级数学公式（类似 $$formula$$ 的模式）
        text = re.sub(r'\$\$([^$]+)\$\$', r'\[\1\]', text)
        
        return text

class MarkdownToLatexConverter:
    """Markdown到LaTeX转换器"""
    
    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager
    
    def convert_to_latex(self, markdown_text: str, filename: str = "document") -> str:
        """将Markdown转换为LaTeX"""
        self.log_manager.add_log("开始Markdown到LaTeX转换", "info")
        
        try:
            # 使用pypandoc进行转换
            self.log_manager.add_log("使用pypandoc进行转换", "info")
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as md_file:
                md_file.write(markdown_text)
                md_file_path = md_file.name
            
            latex_file_path = md_file_path.replace('.md', '.tex')
            
            try:
                # 使用pypandoc转换
                latex_content = pypandoc.convert_file(
                    md_file_path,
                    'latex',
                    format='md',
                    extra_args=['--standalone', '--variable=geometry:margin=1in']
                )
                
                if latex_content:
                    self.log_manager.add_log("pypandoc转换成功", "success")
                    return latex_content
                else:
                    self.log_manager.add_log("pypandoc转换返回空内容", "warning")
                    return self.fallback_conversion(markdown_text, filename)
            
            except Exception as e:
                self.log_manager.add_log(f"pypandoc转换失败: {str(e)}", "error")
                return self.fallback_conversion(markdown_text, filename)
            
            finally:
                # 清理临时文件
                try:
                    os.unlink(md_file_path)
                    if os.path.exists(latex_file_path):
                        os.unlink(latex_file_path)
                except:
                    pass
        
        except Exception as e:
            self.log_manager.add_log(f"转换过程出错: {str(e)}", "error")
            return self.fallback_conversion(markdown_text, filename)
    
    def fallback_conversion(self, markdown_text: str, filename: str) -> str:
        """备用转换方法"""
        self.log_manager.add_log("使用备用转换方法", "info")
        
        # 基础LaTeX文档结构
        latex_content = f"""\\documentclass{{article}}
\\usepackage{{utf8}}
\\usepackage{{geometry}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\geometry{{margin=1in}}

\\title{{{filename.replace('.pdf', '').replace('_', ' ')}}}
\\author{{PDF转换器}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

"""
        
        # 简单的Markdown到LaTeX转换
        lines = markdown_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                latex_content += "\n"
                continue
            
            # 处理标题
            if line.startswith('## '):
                latex_content += f"\\section{{{line[3:]}}}\n\n"
            elif line.startswith('# '):
                latex_content += f"\\section{{{line[2:]}}}\n\n"
            else:
                # 处理数学公式
                line = self.convert_math_formulas(line)
                # 普通段落
                latex_content += f"{line}\\\n\n"
        
        latex_content += "\\end{document}"
        
        self.log_manager.add_log("备用转换完成", "success")
        return latex_content
    
    def convert_math_formulas(self, text: str) -> str:
        """转换数学公式"""
        # 将 \(formula\) 转换为 $formula$
        text = re.sub(r'\\\(([^)]+)\\\)', r'$\1$', text)
        
        # 将 \[formula\] 转换为 $$formula$$
        text = re.sub(r'\\\[([^]]+)\\\]', r'$$\1$$', text)
        
        return text

class LLMOptimizer:
    """大模型优化器"""
    
    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager
    
    def optimize_markdown(self, markdown_text: str, api_key: str, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """使用大模型优化Markdown内容"""
        self.log_manager.add_log(f"开始使用{model}优化Markdown内容", "info")
        
        try:
            import openai
            openai.api_key = api_key
            
            prompt = f"""请优化以下Markdown文档内容，要求：
1. 改善文档结构和格式
2. 修正语法错误和拼写错误
3. 优化数学公式格式（确保使用正确的LaTeX语法）
4. 保持原文的核心内容和意思不变
5. 输出格式必须是有效的Markdown

原始内容：
{markdown_text}

请返回优化后的Markdown内容："""
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的文档编辑和格式化专家，擅长优化Markdown文档和数学公式。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.3
            )
            
            optimized_content = response.choices[0].message.content.strip()
            
            self.log_manager.add_log("大模型优化完成", "success")
            
            return {
                "success": True,
                "optimized_content": optimized_content,
                "original_length": len(markdown_text),
                "optimized_length": len(optimized_content)
            }
        
        except ImportError:
            self.log_manager.add_log("未安装openai库，无法使用大模型优化", "error")
            return {
                "success": False,
                "error": "未安装openai库，请运行: pip install openai"
            }
        except Exception as e:
            self.log_manager.add_log(f"大模型优化失败: {str(e)}", "error")
            return {
                "success": False,
                "error": f"优化失败: {str(e)}"
            }

# 初始化转换器
pdf_converter = PDFConverter(log_manager)
md_to_latex_converter = MarkdownToLatexConverter(log_manager)
llm_optimizer = LLMOptimizer(log_manager)

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/upload_pdf_with_logs', methods=['GET', 'POST'])
def upload_pdf_with_logs():
    """上传PDF文件并实时显示日志"""
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return jsonify({"success": False, "error": "没有选择文件"})
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({"success": False, "error": "没有选择文件"})
            
            if file and file.filename.lower().endswith('.pdf'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                log_manager.add_log(f"文件已保存: {filename}", "info")
                
                # 在后台线程中处理转换
                def convert_pdf():
                    try:
                        log_manager.add_log("开始PDF转换", "info")
                        
                        # 转换PDF为Markdown
                        md_content = pdf_converter.pdf_to_markdown(filepath)
                        
                        if md_content:
                            log_manager.add_log("PDF转换成功", "success")
                            # 这里可以通过某种方式通知前端转换完成
                        else:
                            log_manager.add_log("PDF转换失败", "error")
                    
                    except Exception as e:
                        log_manager.add_log(f"转换过程出错: {str(e)}", "error")
                    
                    finally:
                        # 清理上传的文件
                        try:
                            os.unlink(filepath)
                            log_manager.add_log("临时文件已清理", "info")
                        except:
                            pass
                
                # 启动转换线程
                conversion_thread = threading.Thread(target=convert_pdf)
                conversion_thread.start()
                
                return jsonify({
                    "success": True,
                    "message": "文件上传成功，开始转换",
                    "filename": filename
                })
            
            else:
                return jsonify({"success": False, "error": "只支持PDF文件"})
        
        except Exception as e:
            log_manager.add_log(f"文件上传失败: {str(e)}", "error")
            return jsonify({"success": False, "error": f"上传失败: {str(e)}"})
    
    # GET请求用于Server-Sent Events
    return Response(
        "data: {\"message\": \"连接成功\"}\n\n",
        mimetype="text/event-stream"
    )

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """上传PDF文件"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "没有选择文件"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "没有选择文件"})
        
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            log_manager.add_log(f"文件已保存: {filename}", "info")
            
            # 转换PDF为Markdown
            md_content = pdf_converter.pdf_to_markdown(filepath)
            
            # 清理上传的文件
            try:
                os.unlink(filepath)
                log_manager.add_log("临时文件已清理", "info")
            except:
                pass
            
            if md_content:
                return jsonify({
                    "success": True,
                    "filename": filename,
                    "md_content": md_content
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "PDF转换失败"
                })
        
        else:
            return jsonify({"success": False, "error": "只支持PDF文件"})
    
    except Exception as e:
        log_manager.add_log(f"文件上传失败: {str(e)}", "error")
        return jsonify({"success": False, "error": f"上传失败: {str(e)}"})

@app.route('/convert_to_latex', methods=['POST'])
def convert_to_latex():
    """将Markdown转换为LaTeX"""
    try:
        data = request.get_json()
        md_content = data.get('md_content', '')
        filename = data.get('filename', 'document')
        optimize_with_llm = data.get('optimize_with_llm', False)
        llm_api_key = data.get('llm_api_key', '')
        
        if not md_content:
            return jsonify({"success": False, "error": "没有Markdown内容"})
        
        # 如果需要大模型优化
        if optimize_with_llm and llm_api_key:
            optimization_result = llm_optimizer.optimize_markdown(
                md_content, llm_api_key
            )
            
            if optimization_result["success"]:
                md_content = optimization_result["optimized_content"]
                log_manager.add_log("大模型优化成功", "success")
            else:
                log_manager.add_log(f"大模型优化失败: {optimization_result.get('error', '未知错误')}", "warning")
        
        # 转换为LaTeX
        latex_content = md_to_latex_converter.convert_to_latex(md_content, filename)
        
        if latex_content:
            return jsonify({
                "success": True,
                "latex_content": latex_content
            })
        else:
            return jsonify({
                "success": False,
                "error": "LaTeX转换失败"
            })
    
    except Exception as e:
        log_manager.add_log(f"LaTeX转换失败: {str(e)}", "error")
        return jsonify({"success": False, "error": f"转换失败: {str(e)}"})

@app.route('/optimize_md', methods=['POST'])
def optimize_md():
    """优化Markdown内容"""
    try:
        data = request.get_json()
        md_content = data.get('md_content', '')
        llm_api_key = data.get('llm_api_key', '')
        model = data.get('model', 'gpt-3.5-turbo')
        
        if not md_content:
            return jsonify({"success": False, "error": "没有Markdown内容"})
        
        if not llm_api_key:
            return jsonify({"success": False, "error": "没有提供API密钥"})
        
        result = llm_optimizer.optimize_markdown(md_content, llm_api_key, model)
        
        return jsonify(result)
    
    except Exception as e:
        log_manager.add_log(f"Markdown优化失败: {str(e)}", "error")
        return jsonify({"success": False, "error": f"优化失败: {str(e)}"})

@app.route('/download_latex', methods=['POST'])
def download_latex():
    """下载LaTeX文件"""
    try:
        data = request.get_json()
        latex_content = data.get('latex_content', '')
        filename = data.get('filename', 'document.tex')
        
        if not latex_content:
            return jsonify({"success": False, "error": "没有LaTeX内容"})
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False, encoding='utf-8') as tex_file:
            tex_file.write(latex_content)
            tex_file_path = tex_file.name
        
        log_manager.add_log(f"生成LaTeX文件: {filename}", "info")
        
        return send_file(
            tex_file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/x-tex'
        )
    
    except Exception as e:
        log_manager.add_log(f"LaTeX文件下载失败: {str(e)}", "error")
        return jsonify({"success": False, "error": f"下载失败: {str(e)}"})

@app.route('/get_latest_logs', methods=['GET'])
def get_latest_logs():
    """获取最新的日志"""
    try:
        logs = log_manager.get_logs(50)  # 获取最近50条日志
        return jsonify({"logs": logs})
    except Exception as e:
        log_manager.add_log(f"获取日志失败: {str(e)}", "error")
        return jsonify({"logs": []})

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    """清空日志"""
    try:
        log_manager.clear_logs()
        log_manager.add_log("日志已清空", "info")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.errorhandler(413)
def too_large(e):
    """文件太大错误处理"""
    log_manager.add_log("上传文件太大", "error")
    return jsonify({"success": False, "error": "文件大小不能超过16MB"}), 413

@app.errorhandler(500)
def internal_error(e):
    """内部服务器错误处理"""
    log_manager.add_log(f"内部服务器错误: {str(e)}", "error")
    return jsonify({"success": False, "error": "内部服务器错误"}), 500

if __name__ == '__main__':
    log_manager.add_log("启动PDF转Markdown转LaTeX服务", "info")
    print("PDF转Markdown转LaTeX服务启动中...")
    print("请访问: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)