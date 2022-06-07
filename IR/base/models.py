from django.db import models

# Create your models here.
# data_set_1
class DataSet1(models.Model):
    abstract = models.TextField(null=True)
    
    def __str__(self):
        return str(self.pk)
    
class references1(models.Model):
    doc1 = models.IntegerField()
    degree = models.IntegerField()
    doc2 = models.ForeignKey(DataSet1, on_delete=models.CASCADE , null=True)
    
    def __str__(self):
        return str(self.doc2.pk)
    
class QuerySet1(models.Model):
    query = models.TextField(null=True)
    def __str__(self):
        return self.query
    
class Query_Document1(models.Model):
    query_ID = models.IntegerField()
    document_ID = models.IntegerField()
    def __str__(self):
        return str(self.query_ID)
    
