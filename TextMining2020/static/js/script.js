(function ($, window) {
	
	new TableExport($('table'), {formats: [ 'xls', 'csv'], fileName: "contact-list", bootstrap: true})
	//newTableExport($('table'), {formats: ['xlsx', 'xls', 'csv', 'txt'], fileName: "contact-list", bootstrap: true})

}).call(this, jQuery, window);

