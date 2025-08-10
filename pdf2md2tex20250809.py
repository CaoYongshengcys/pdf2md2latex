import os
import subprocess
import re
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tempfile
import pypandoc

# ... (Flask app 初始化等代码保持不变) ...
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def convert_pdf_to_md(pdf_path, output_dir):
    """
    使用MinerU将PDF转换为Markdown文件。
    返回一个元组 (is_success, result_path_or_error_message)。
    """
    try:
        # --- 核心修改点 ---
        # 1. 增加了 '-y' 标志，尝试自动确认所有提示，防止交互式等待。
        #    如果 '-y' 不是正确的标志，您需要查阅 MinerU 的文档换成正确的。
        # 2. 您的代码中有一个小笔误，'-p' 标志应该直接跟在路径前，而不是作为独立参数。
        #    但根据您的写法，我假设它的用法是 ['...mineru.exe', '-p', 'path']
        cmd = [
            r'C:\Users\cys2025\.conda\envs\stategrid\Scripts\mineru.exe',
            '-y',  # 非交互模式
            '--page-by-page',  # 每页进行转换
            '--no-header-footer',  # 不识别页眉页脚
            '--skip-ocr',  # 跳过OCR处理，提高速度
            '--disable-tables',  # 禁用表格识别，减少复杂度
            '--disable-formula',  # 禁用公式识别，减少复杂度
            '-p', pdf_path,
            '-o', output_dir
        ]

        print(f"Executing command with timeout: {' '.join(cmd)}")

        # 3. 增加了 timeout=60 参数。如果进程60秒内未完成，将抛出 TimeoutExpired 异常。
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=300  # 增加到300秒超时，适应复杂文档
        )

        # --- 后续逻辑与上一版类似，但现在是在有超时保障的情况下执行 ---
        print("--- MinerU STDOUT ---")
        print(result.stdout if result.stdout.strip() else "[EMPTY]")
        print("--- MinerU STDERR ---")
        print(result.stderr if result.stderr.strip() else "[EMPTY]")
        print(f"--- MinerU Return Code: {result.returncode} ---")
        
        if result.returncode != 0:
            error_message = f"MinerU exited with error code {result.returncode}.\n---STDERR---\n{result.stderr}\n---STDOUT---\n{result.stdout}"
            return False, error_message

        # 查找输出文件
        found_file_path = None
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        files_in_dir = os.listdir(output_dir)

        for file in files_in_dir:
            if file.endswith('.md'):
                found_file_path = os.path.join(output_dir, file)
                break
        
        if found_file_path:
            return True, found_file_path

        # 如果找不到文件，检查stdout
        if result.stdout and len(result.stdout.strip()) > 0:
            output_md_path = os.path.join(output_dir, f"{base_name}_from_stdout.md")
            with open(output_md_path, 'w', encoding='utf-8') as f:
                f.write(result.stdout)
            return True, output_md_path

        return False, "MinerU ran successfully, but no Markdown file was found and STDOUT was empty."

    except FileNotFoundError:
        return False, "Error: 'mineru.exe' command not found. Please check the hardcoded path in the script."
    except subprocess.TimeoutExpired:
        # 捕获超时异常，并返回清晰的错误信息
        return False, "Error: PDF conversion timed out after 60 seconds. The PDF may be too large, complex, or the process is stuck."
    except Exception as e:
        return False, f"An unexpected error occurred during PDF conversion: {str(e)}"

# ... (文件的其余部分，包括所有 Flask 路由，都保持不变) ...
def convert_md_to_latex(md_content):
    """
    使用 pypandoc 将Markdown内容转换为LaTeX格式。
    返回一个元组 (is_success, latex_content_or_error_message)。
    """
    try:
        # 使用pypandoc进行转换，它比正则表达式健壮得多
        latex_content = pypandoc.convert_text(md_content, 'latex', format='md')
        return True, latex_content
    except Exception as e:
        error_msg = f"Pandoc conversion to LaTeX failed: {str(e)}"
        print(error_msg)
        return False, error_msg

@app.route('/')
def index():
    """主页面"""
    # 我们将HTML直接放在这里，方便作为一个单独文件运行
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """上传PDF文件并转换为Markdown"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)
        
        # 创建一个临时输出目录用于存放转换结果
        with tempfile.TemporaryDirectory() as temp_dir:
            success, result = convert_pdf_to_md(pdf_path, temp_dir)
            
            if success:
                md_file_path = result
                with open(md_file_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                
                return jsonify({
                    'success': True,
                    'md_content': md_content,
                    'filename': filename
                })
            else:
                # 返回详细的错误信息
                return jsonify({'error': f'PDF to Markdown conversion failed:\n{result}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Only PDF is supported.'}), 400

@app.route('/convert_to_latex', methods=['POST'])
def convert_to_latex_route():
    """将前端发来的Markdown内容转换为LaTeX"""
    data = request.get_json()
    if not data or 'md_content' not in data:
        return jsonify({'error': 'No Markdown content provided'}), 400
    
    md_content = data['md_content']
    
    success, result = convert_md_to_latex(md_content)
    
    if success:
        return jsonify({
            'success': True,
            'latex_content': result
        })
    else:
        # 返回详细的错误信息
        return jsonify({'error': f'Markdown to LaTeX conversion failed:\n{result}'}), 500

@app.route('/download_latex', methods=['POST'])
def download_latex():
    """下载LaTeX文件"""
    data = request.get_json()
    if not data or 'latex_content' not in data:
        return jsonify({'error': 'No LaTeX content provided'}), 400
    
    latex_content = data['latex_content']
    filename = data.get('filename', 'output.tex')
    # 确保文件名以.tex结尾
    base_name = os.path.splitext(secure_filename(filename))[0]
    download_name = f"{base_name}.tex"

    # 创建一个临时文件来保存内容
    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.tex', encoding='utf-8') as temp_file:
            temp_file.write(latex_content)
            temp_path = temp_file.name
        
        return send_file(temp_path, as_attachment=True, download_name=download_name)
    finally:
        # 确保在发送文件后删除临时文件
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == '__main__':
    # 确保在程序退出时清理上传文件夹
    import atexit
    import shutil
    atexit.register(lambda: shutil.rmtree(app.config['UPLOAD_FOLDER']))
    app.run(debug=True)