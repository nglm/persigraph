from copyreg import constructor
from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def main(request):
    return render(request, 'index.html', {"app" : "MJO"})

def relevant(request):
    print(request, flush=True)
    print("testing")
    # g = request.GET['g']
    # selected_k = request.GET["k"]
    # vertices, edges = g.get_relevant_components(selected_k)
    # return HttpResponse({"vertices" : vertices, "edges" : edges})
    return HttpResponse({"vertices" : "vertices", "edges" : "edges"})