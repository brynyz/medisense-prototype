$(document).ready(function() {
  $('[data-toggle="tooltip"]').tooltip();

  // Open popup for Add or Edit
  $(".popup-btn").on("click", function(e) {
    e.preventDefault();
    $(".popup").addClass("active");

    // Check if this is an edit button
    var editUrl = $(this).data("edit-url");
    var itemId = $(this).data("item-id");
    
    if (editUrl && itemId) {
      // This is an EDIT operation
      $(".popup-form").attr("action", editUrl);
      $(".btn-text").text("Update Item");
      
      // TODO: Load item data and populate form fields
      // You can make an AJAX call here to get item details and populate the form
      
    } else {
      // This is an ADD operation
      $(".popup-form").attr("action", "/inventory/add/");
      $(".btn-text").text("Add Item");
      
      // Reset form for Add
      $(".popup-form")[0].reset();
      
      // Reset all select boxes to default "Select..." option
      $("#category").val("").prop('selectedIndex', 0);
      $("#unit").val("").prop('selectedIndex', 0);
      $("#status").val("").prop('selectedIndex', 0);
    }
  });

  // Close popup
  $(".popup .close-btn").on("click", function() {
    $(".popup").removeClass("active");
  });

  // Close popup when clicking outside
  $(".popup").on("click", function(e) {
    if (e.target === this) {
      $(".popup").removeClass("active");
    }
  });

  // AJAX form submission
  $(".popup-form").on("submit", function(e) {
    e.preventDefault();
    
    var form = $(this);
    var formData = new FormData(form[0]);
    var url = form.attr('action');

    // Check if URL is set
    if (!url) {
      alert('Error: Form action not set properly');
      return;
    }

    $.ajax({
      url: url,
      type: 'POST',
      data: formData,
      processData: false,
      contentType: false,
      headers: {
        'X-CSRFToken': $('[name=csrfmiddlewaretoken]').val(),
        'X-Requested-With': 'XMLHttpRequest'
      },
      success: function(response) {
        console.log("AJAX Success:", response);
        if (response.success) {
          // Close the popup
          $(".popup").removeClass("active");
          
          // Show success message
          alert(response.message || 'Operation successful!');
          
          // Reload the page content
          location.reload();
        } else {
          // Handle form errors
          alert('Error: ' + (response.error || 'Please check your input'));
        }
      },
      error: function(xhr, status, error) {
        console.error('AJAX Error:', xhr.responseText);
        
        try {
          var response = JSON.parse(xhr.responseText);
          alert('Error: ' + (response.error || 'Please check your input'));
        } catch(e) {
          alert('An error occurred. Please try again. Status: ' + xhr.status);
        }
      }
    });
  });

  // AJAX delete functionality
  $(document).on('click', '.delete-btn', function(e) {
    e.preventDefault();
    
    if (confirm('Are you sure you want to delete this item?')) {
      var deleteUrl = $(this).attr('href');
      var row = $(this).closest('tr');
      
      $.ajax({
        url: deleteUrl,
        type: 'POST',
        headers: {
          'X-CSRFToken': $('[name=csrfmiddlewaretoken]').val(),
          'X-Requested-With': 'XMLHttpRequest'
        },
        success: function(response) {
          if (response.success) {
            // Remove the row with animation
            row.fadeOut(300, function() {
              $(this).remove();
              
              // Check if table is empty
              if ($('tbody tr').length === 0) {
                $('tbody').html('<tr><td colspan="7" class="text-center">No items in inventory</td></tr>');
              }
            });
            
            alert(response.message || 'Item deleted');
          } else {
            alert('Error deleting item: ' + (response.error || 'Unknown error'));
          }
        },
        error: function(xhr, status, error) {
          console.error('Delete Error:', xhr.responseText);
          alert('An error occurred while deleting. Please try again.');
        }
      });
    }
  });
});