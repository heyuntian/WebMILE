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
	if request.method == 'POST':
		try:
			fs = FileSystemStorage()
			graph = request.FILES['graph']
			graph_name = fs.save(graph.name, graph)
			script = request.FILES['script']
			script_name = fs.save(script.name, script)
			others = request.FILES['others']
			others_name = fs.save(others.name, others)
			data["dim"] =  request.POST['dim']
			data["coarsen"] =  request.POST['coarsen']
			data['comm'] = request.POST['comm']
			data["language"] = request.POST['language']
			# print(graph, graph_name, graph.name)
			# print(type(graph), type(graph_name), type(graph.name))
			print("file save")
			print(data)

			# Move data to backend
			jobid = random.randint(0, 1000000)  # random token
			des = f'backend/jobs/{jobid}'
			des_src = os.path.join(des, 'src')
			if not os.path.exists(des):
				os.mkdir(des)
			if not os.path.exists(des_src):
				os.mkdir(des_src)
			in_format = getExtension(graph_name)
			os.rename(f'media/{graph_name}', os.path.join(des, f'graph.{in_format}'))
			ext_script = getExtension(script_name)
			if ext_script == 'zip':
				os.system(f'mv media/{script_name} {des}')
				os.system(f'unzip {des}/{script_name} -d {des_src}')
				os.system(f'rm {des}/{script_name}')
			else:
				os.rename(f'media/{script_name}', os.path.join(des_src, script))  # Here we use the original name for the second argument
			os.rename(f'media/{others_name}', os.path.join(des, 'requirements.txt'))
			os.system(f'rm media/{graph_name} media/{script_name} media/{others_name}')

			# Run backend
			os.system('conda activate MILE-interface')
			msg["msg"] = "msg: Starting running mile!"
			root = f'backend/jobs'
			out_format = 'edgelist'
			coarsen_level = int(data['coarsen'])
			embed_dim = int(data['dim'])
			language = 'python'
			arguments = data['comm']
			os.system(f'python backend/main_API.py --root {root} --jobid {jobid} --in-format {in_format} --out-format {out_format} --coarsen-level {coarsen_level} --embed-dim {embed_dim} --language {language} --arguments "{arguments}"')
			msg["msg"] = 'Available to download'
			msg["finish"]= True
			embedding_path, new_path = f"backend/jobs/{jobid}/embeddings.txt", f'media/{jobid}_embeddings.txt'
			os.rename(embedding_path, new_path)
			msg['download_link'] = new_path

		except Exception:
			msg["msg"] = "msg: Please upload required files"
	return render(request, 'upload.html',msg)


