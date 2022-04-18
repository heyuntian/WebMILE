from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, CreateView
from django.core.files.storage import FileSystemStorage
from django.urls import reverse_lazy
from django.core import management
import os
import random

class Home(TemplateView):
    template_name = 'upload.html'
data = {}


def getExtension(filename):
	return filename.split('.')[-1]


def upload(request):
	msg = {}
	msg["finish"]= False
	msg["alert"] = False
	if request.method == 'POST':
		try:
			msg["alert"] =True
			fs = FileSystemStorage()
			graph = request.FILES['graph']
			graph_name = fs.save(graph.name, graph)
			msg["graph"] = graph.name
			data["dim"] =  request.POST['dim']
			data["coarsen"] =  request.POST['coarsen']
			exist_embed = data['exist_embedding'] = request.POST['embedding'].lower()
			if exist_embed == 'none':
				script = request.FILES['script']
				script_name = fs.save(script.name, script)
				msg["script"] = script.name
				data['comm'] = request.POST['comm']
				data["language"] = request.POST['language'].lower()
			else:
				# script = request.FILES['script']
				# script_name = fs.save(script.name, script)
				# msg["script"] = graph.name
				data['comm'] = request.POST['comm']
				# TODO: Make uploading optional. 
				if exist_embed == 'line':
					data['language'] = 'r'
				elif exist_embed == 'node2vec':
					data['language'] = 'java'
				else:
					data['language'] = 'python'
				ext = 'jar' if data['language'] == 'java' else 'zip'
				script_name = f'{exist_embed}_{data["language"]}.{ext}'
			
			# print(graph, graph_name, graph.name)
			# print(type(graph), type(graph_name), type(graph.name))
			print("file save")
			print(data)
			render(request, 'upload.html', msg)

			# Move graph data to backend
			jobid = random.randint(0, 1000000)  # random token
			des = f'backend/jobs/{jobid}'
			des_src = os.path.join(des, 'src')
			if not os.path.exists(des):
				os.mkdir(des)
			if not os.path.exists(des_src):
				os.mkdir(des_src)
			in_format = getExtension(graph_name)
			os.rename(f'media/{graph_name}', os.path.join(des, f'graph.{in_format}'))
			print('move graph')
			# os.system(f'rm media/{graph_name}')

			# Move uploaded or provided embedding method to backend
			if exist_embed != 'none':
				# User selects a provided method
				# os.system(f'rm media/{script_name}')
				os.system(f'cp examples/{script_name} {des}')
			else:
				os.system(f'mv media/{script_name} {des}')

			ext_script = getExtension(script_name)
			if ext_script == 'zip':
				os.system(f'unzip {des}/{script_name} -d {des_src}')
				os.system(f'rm {des}/{script_name}')
			else:
				os.rename(f'{des}/{script_name}', os.path.join(des_src, f'embed.{ext_script}'))  # Here we use the original name for the second argument
			print('move script')
			

			# Run backend
			# os.system('conda init bash')
			os.system('conda activate MILE-interface')
			msg["msg"] = "msg: Starting running mile!"
			root = f'backend/jobs'
			out_format = 'edgelist'
			coarsen_level = int(data['coarsen'])
			embed_dim = int(data['dim'])
			language = data['language']
			arguments = data['comm']
			os.system(f'python backend/main_API.py --root {root} --jobid {jobid} --in-format {in_format} --out-format {out_format} --coarsen-level {coarsen_level} --embed-dim {embed_dim} --language {language}  --arguments "{arguments}"')
			msg["msg"] = 'Available to download'
			msg["finish"]= True
			embedding_path, new_path = f"backend/jobs/{jobid}/embeddings.txt", f'media/{jobid}_embeddings.txt'
			os.rename(embedding_path, new_path)
			os.system(f'rm -R {des}')
			msg['download_link'] = new_path

		except Exception:
			msg["msg"] = "msg: Please upload correct files"
	return render(request, 'upload.html',msg)


# from django.shortcuts import render, redirect
# from django.views.generic import TemplateView, ListView, CreateView
# from django.core.files.storage import FileSystemStorage
# from django.urls import reverse_lazy
# from django.core import management
# import os
# import random

# class Home(TemplateView):
#     template_name = 'upload.html'
# data = {}


# def getExtension(filename):
# 	return filename.split('.')[-1]


# def upload(request):
# 	msg = {}
# 	msg["finish"]= False
# 	if request.method == 'POST':
# 		try:
# 			fs = FileSystemStorage()
# 			graph = request.FILES['graph']
# 			graph_name = fs.save(graph.name, graph)
# 			script = request.FILES['script']
# 			script_name = fs.save(script.name, script)
# 			# others = request.FILES['others']
# 			# others_name = fs.save(others.name, others)
# 			data["dim"] =  request.POST['dim']
# 			data["coarsen"] =  request.POST['coarsen']
# 			data['comm'] = request.POST['comm']
# 			data["language"] = request.POST['language'].lower()
# 			# print(graph, graph_name, graph.name)
# 			# print(type(graph), type(graph_name), type(graph.name))
# 			print("file save")
# 			print(data)

# 			# Move data to backend
# 			jobid = random.randint(0, 1000000)  # random token
# 			des = f'backend/jobs/{jobid}'
# 			des_src = os.path.join(des, 'src')
# 			if not os.path.exists(des):
# 				os.mkdir(des)
# 			if not os.path.exists(des_src):
# 				os.mkdir(des_src)
# 			in_format = getExtension(graph_name)
# 			os.rename(f'media/{graph_name}', os.path.join(des, f'graph.{in_format}'))
# 			print('move graph')
# 			ext_script = getExtension(script_name)
# 			if ext_script == 'zip':
# 				os.system(f'mv media/{script_name} {des}')
# 				os.system(f'unzip {des}/{script_name} -d {des_src}')
# 				os.system(f'rm {des}/{script_name}')
# 			else:
# 				os.rename(f'media/{script_name}', os.path.join(des_src, f'embed.{ext_script}'))  # Here we use the original name for the second argument
# 			print('move script')

# 			req_file_name = None
# 			if data['language'] == 'python':
# 				req_file_name = 'requirements.txt'
# 			elif data['language'] == 'r':
# 				req_file_name = 'install_packages.R'
# 			# os.rename(f'media/{others_name}', os.path.join(des, req_file_name))
# 			# os.system(f'rm media/{graph_name} media/{script_name} media/{others_name}')

# 			# Run backend
# 			os.system('conda activate MILE-interface')
# 			msg["msg"] = "msg: Starting running mile!"
# 			root = f'backend/jobs'
# 			out_format = 'edgelist'
# 			coarsen_level = int(data['coarsen'])
# 			embed_dim = int(data['dim'])
# 			language = data['language']
# 			arguments = data['comm']
# 			os.system(f'python backend/main_API.py --root {root} --jobid {jobid} --in-format {in_format} --out-format {out_format} --coarsen-level {coarsen_level} --embed-dim {embed_dim} --language {language}  --arguments "{arguments}"')
# 			msg["msg"] = 'Available to download'
# 			msg["finish"]= True
# 			embedding_path, new_path = f"backend/jobs/{jobid}/embeddings.txt", f'media/{jobid}_embeddings.txt'
# 			os.rename(embedding_path, new_path)
# 			msg['download_link'] = new_path

# 		except Exception:
# 			msg["msg"] = "msg: Please upload required files"
# 	return render(request, 'upload.html',msg)